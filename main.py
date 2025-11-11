import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, neighbor_channels=24, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        pos_emb_dim = 16
        self.position_embeddings = nn.Parameter(torch.randn(8, pos_emb_dim) * 0.02)
        
        self.pos_projection = nn.Sequential(
            nn.Linear(pos_emb_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 3),
            nn.Tanh()
        )

        total_in = in_channels + neighbor_channels + 8
        
        self.enc1 = self.conv_block(total_in, 64, time_emb_dim)
        self.enc2 = self.conv_block(64, 128, time_emb_dim)
        self.enc3 = self.conv_block(128, 256, time_emb_dim)
        self.bottleneck = self.conv_block(256, 512, time_emb_dim)
        self.dec3 = self.conv_block(512 + 256, 256, time_emb_dim)
        self.dec2 = self.conv_block(256 + 128, 128, time_emb_dim)
        self.dec1 = self.conv_block(128 + 64, 64, time_emb_dim)
        self.final = nn.Conv2d(64, out_channels, 1)

    def conv_block(self, in_ch, out_ch, time_emb_dim):
        return nn.ModuleDict({
            'conv1': nn.Conv2d(in_ch, out_ch, 3, padding=1),
            'conv2': nn.Conv2d(out_ch, out_ch, 3, padding=1),
            'time_emb': nn.Linear(time_emb_dim, out_ch),
            'norm1': nn.GroupNorm(8, out_ch),
            'norm2': nn.GroupNorm(8, out_ch),
        })

    def forward_conv_block(self, x, block, time_emb, pool=True):
        x = block['conv1'](x)
        x = block['norm1'](x)
        x = F.silu(x)

        time_emb_proj = block['time_emb'](time_emb)[:, :, None, None]
        x = x + time_emb_proj
        x = block['conv2'](x)
        x = block['norm2'](x)
        x = F.silu(x)
        if pool:
            return F.max_pool2d(x, 2), x
        return x

    def forward(self, x, t, neighbors, neighbor_mask=None):
        B, num_neighbors, C, H, W = neighbors.shape

        if neighbor_mask is None:
            neighbor_mask = torch.ones(B, num_neighbors, device=x.device)

        neighbors_with_pos = []
        validity_channels = []

        for i in range(num_neighbors):
            pos_emb = self.position_embeddings[i]
            pos_emb = self.pos_projection(pos_emb)
            pos_emb = pos_emb.view(1, 3, 1, 1).expand(B, 3, H, W)
            
            neighbor_with_pos = neighbors[:, i] + pos_emb * 0.1

            mask = neighbor_mask[:, i].view(B, 1, 1, 1)
            neighbor_with_pos = neighbor_with_pos * mask

            neighbors_with_pos.append(neighbor_with_pos)

            validity_channel = mask.expand(B, 1, H, W)
            validity_channels.append(validity_channel)

        neighbors_encoded = torch.stack(neighbors_with_pos, dim=1)
        validity_encoded = torch.stack(validity_channels, dim=1)
        neighbors_flat = neighbors_encoded.reshape(B, -1, H, W)
        validity_flat = validity_encoded.reshape(B, -1, H, W)
        x_combined = torch.cat([x, neighbors_flat, validity_flat], dim=1)
        time_emb = self.time_mlp(t.float().view(-1, 1))
        x1, skip1 = self.forward_conv_block(x_combined, self.enc1, time_emb, pool=True)
        x2, skip2 = self.forward_conv_block(x1, self.enc2, time_emb, pool=True)
        x3, skip3 = self.forward_conv_block(x2, self.enc3, time_emb, pool=True)
        x_bottle = self.forward_conv_block(x3, self.bottleneck, time_emb, pool=False)

        x = F.interpolate(x_bottle, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip3], dim=1)
        x = self.forward_conv_block(x, self.dec3, time_emb, pool=False)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)
        x = self.forward_conv_block(x, self.dec2, time_emb, pool=False)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)
        x = self.forward_conv_block(x, self.dec1, time_emb, pool=False)

        return self.final(x)


class TextureTileDataset(Dataset):
    def __init__(self, image_path, tile_size=64, num_samples=500, overlap=8,
                 missing_neighbor_prob=0.3, force_missing_patterns=True):
        self.image = Image.open(image_path).convert('RGB')
        self.img_array = np.array(self.image)
        self.tile_size = tile_size
        self.num_samples = num_samples
        self.overlap = overlap
        self.missing_neighbor_prob = missing_neighbor_prob
        self.force_missing_patterns = force_missing_patterns
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # Разного рода паттерны отсутствующих соседей сгенерировал с Claude
        self.missing_patterns = [
            # Single cross neighbor available
            [0, 2, 3, 4, 5, 7],  # Only top neighbor
            [0, 1, 2, 4, 5, 7],  # Only left neighbor
            [0, 1, 2, 3, 5, 6],  # Only right neighbor
            [0, 1, 2, 3, 4, 5],  # Only bottom neighbor

            # Complete sides missing
            [3, 0, 5],  # Complete left side
            [4, 2, 7],  # Complete right side
            [1, 0, 2],  # Complete top side
            [6, 5, 7],  # Complete bottom side

            # Corner combinations
            [0, 1, 2, 3, 4],  # Top + both horizontal sides (bottom missing)
            [5, 6, 7, 3, 4],  # Bottom + both horizontal sides (top missing)
            [0, 1, 3, 5, 6],  # Left + both vertical sides (right missing)
            [2, 4, 1, 6, 7],  # Right + both vertical sides (left missing)

            # Specific corners
            [0, 1, 3],  # Top-left corner
            [1, 2, 4],  # Top-right corner
            [3, 5, 6],  # Bottom-left corner
            [4, 6, 7],  # Bottom-right corner

            # Diagonal patterns
            [0, 2, 5, 7],  # All corners
            [1, 3, 4, 6],  # All sides (no corners)

            # Sparse patterns (very few neighbors)
            [0, 1, 2, 3, 5, 6, 7],  # Only right neighbor
            [0, 1, 2, 4, 5, 6, 7],  # Only left neighbor
            [0, 2, 3, 4, 5, 7],  # Only top and bottom
            [1, 6],  # Only left and right sides

            # No neighbors at all (extreme case)
            list(range(8)),  # All missing
        ]

        if self.force_missing_patterns:
            self.samples_per_pattern = max(1, num_samples // (len(self.missing_patterns) * 2))

    def __len__(self):
        return self.num_samples

    def get_tile(self, y, x):
        h, w = self.img_array.shape[:2]
        tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)

        for i in range(self.tile_size):
            for j in range(self.tile_size):
                src_y = (y + i) % h
                src_x = (x + j) % w
                tile[i, j] = self.img_array[src_y, src_x]

        return self.transform(Image.fromarray(tile))

    def get_missing_pattern(self, idx):
        if not self.force_missing_patterns:
            return None

        pattern_samples = self.samples_per_pattern * len(self.missing_patterns)

        if idx < pattern_samples:
            pattern_idx = idx // self.samples_per_pattern
            pattern_idx = pattern_idx % len(self.missing_patterns)
            return self.missing_patterns[pattern_idx]
        else:
            return None

    def __getitem__(self, idx):
        h, w = self.img_array.shape[:2]

        center_y = random.randint(0, h - 1)
        center_x = random.randint(0, w - 1)

        center_tile = self.get_tile(center_y, center_x)

        positions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0), (1, 1)
        ]

        neighbors = []
        neighbor_mask = []

        missing_indices = self.get_missing_pattern(idx)

        for i, (dy, dx) in enumerate(positions):
            neighbor_y = center_y + dy * (self.tile_size - self.overlap)
            neighbor_x = center_x + dx * (self.tile_size - self.overlap)
            neighbor_tile = self.get_tile(neighbor_y, neighbor_x)

            if missing_indices is not None:
                is_valid = i not in missing_indices
            else:
                is_valid = random.random() > self.missing_neighbor_prob

            if is_valid:
                neighbors.append(neighbor_tile)
                neighbor_mask.append(1.0)
            else:
                neighbors.append(torch.zeros_like(neighbor_tile))
                neighbor_mask.append(0.0)

        neighbors = torch.stack(neighbors)
        neighbor_mask = torch.tensor(neighbor_mask, dtype=torch.float32)

        return center_tile, neighbors, neighbor_mask


class DDPMDiffusion:
    def __init__(self, model, timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]).to(device),
            self.alphas_cumprod[:-1]
        ])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return xt, noise

    @torch.no_grad()
    def sample(self, neighbors, neighbor_mask, tile_size=64, color_guidance_scale=0.3):
        self.model.eval()
        
        valid_neighbors = neighbors[neighbor_mask > 0.5]
        if len(valid_neighbors) > 0:
            target_mean = valid_neighbors.mean(dim=[0, 2, 3])
            target_std = valid_neighbors.std(dim=[0, 2, 3])
        else:
            target_mean = torch.zeros(3, device=neighbors.device)
            target_std = torch.ones(3, device=neighbors.device)
        
        x = torch.randn(1, 3, tile_size, tile_size).to(device)
        neighbors = neighbors.unsqueeze(0).to(device)
        neighbor_mask = neighbor_mask.unsqueeze(0).to(device)

        for t in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling", leave=False):
            t_batch = torch.tensor([t], device=device)

            predicted_noise = self.model(x, t_batch, neighbors, neighbor_mask)

            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            if t > 0:
                alpha_cumprod_t_prev = self.alphas_cumprod[t - 1]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0).to(device)

            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)

            if len(valid_neighbors) > 0 and t > self.timesteps // 2:
                current_mean = pred_x0.mean(dim=[2, 3])
                current_std = pred_x0.std(dim=[2, 3])
            
                correction = (target_mean - current_mean) * color_guidance_scale
                pred_x0 = pred_x0 + correction.view(1, 3, 1, 1)

            if t == 0:
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            posterior_variance = beta_t * (1.0 - alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t)
            posterior_mean_coef1 = beta_t * torch.sqrt(alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t)
            posterior_mean_coef2 = (1.0 - alpha_cumprod_t_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_cumprod_t)

            posterior_mean = posterior_mean_coef1 * pred_x0 + posterior_mean_coef2 * x

            if t > 0:
                noise = torch.randn_like(x)
                x = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                x = posterior_mean

        return x


def train_diffusion(image_path, tile_size=64, epochs=50, batch_size=8, lr=0.0001,
                    num_samples=500, patience=15, min_delta=0.0005,
                    missing_neighbor_prob=0.3, force_missing_patterns=True,
                    timesteps=100, beta_start=0.0001, beta_end=0.02):
    total_samples = num_samples
    train_samples = int(total_samples * 0.85)
    val_samples = total_samples - train_samples

    train_dataset = TextureTileDataset(
        image_path, tile_size=tile_size, num_samples=train_samples,
        missing_neighbor_prob=missing_neighbor_prob,
        force_missing_patterns=force_missing_patterns
    )
    val_dataset = TextureTileDataset(
        image_path, tile_size=tile_size, num_samples=val_samples,
        missing_neighbor_prob=missing_neighbor_prob,
        force_missing_patterns=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet(in_channels=3, out_channels=3, neighbor_channels=24).to(device)
    diffusion = DDPMDiffusion(model, timesteps=timesteps, beta_start=beta_start, beta_end=beta_end)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for center_tiles, neighbors, neighbor_mask in pbar:
            center_tiles = center_tiles.to(device)
            neighbors = neighbors.to(device)
            neighbor_mask = neighbor_mask.to(device)

            t = torch.randint(0, diffusion.timesteps, (center_tiles.shape[0],), device=device)

            noisy_tiles, noise = diffusion.forward_diffusion(center_tiles, t)

            predicted_noise = model(noisy_tiles, t, neighbors, neighbor_mask)

            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for center_tiles, neighbors, neighbor_mask in val_loader:
                center_tiles = center_tiles.to(device)
                neighbors = neighbors.to(device)
                neighbor_mask = neighbor_mask.to(device)

                t = torch.randint(0, diffusion.timesteps, (center_tiles.shape[0],), device=device)
                noisy_tiles, noise = diffusion.forward_diffusion(center_tiles, t)
                predicted_noise = model(noisy_tiles, t, neighbors, neighbor_mask)

                loss = criterion(predicted_noise, noise)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
           

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    diffusion_config = {
        'timesteps': timesteps,
        'beta_start': beta_start,
        'beta_end': beta_end
    }

    return model, diffusion, diffusion_config, train_dataset, val_dataset

def synthesize_3x3_texture_checkerboard(model, diffusion, original_image_path, tile_size=64, overlap=8):
    model.eval()
    device = next(model.parameters()).device

    original = Image.open(original_image_path).convert('RGB')
    orig_w, orig_h = original.size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    grid_h_orig = (orig_h + tile_size - overlap - 1) // (tile_size - overlap)
    grid_w_orig = (orig_w + tile_size - overlap - 1) // (tile_size - overlap)

    grid_h = grid_h_orig * 3
    grid_w = grid_w_orig * 3

    print(f"Original grid size: {grid_h_orig}x{grid_w_orig} tiles")
    print(f"Target grid size: {grid_h}x{grid_w} tiles")

    output_h = grid_h * (tile_size - overlap) + overlap
    output_w = grid_w * (tile_size - overlap) + overlap
    canvas = torch.zeros(3, output_h, output_w).to(device)
    weight_map = torch.zeros(1, output_h, output_w).to(device)

    generated_tiles = {}
    
    original_tiles_list = []

    original_array = np.array(original)

    for i in range(grid_h_orig):
        for j in range(grid_w_orig):
            tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            for ti in range(tile_size):
                for tj in range(tile_size):
                    src_y = (i * (tile_size - overlap) + ti) % orig_h
                    src_x = (j * (tile_size - overlap) + tj) % orig_w
                    tile[ti, tj] = original_array[src_y, src_x]

            tile_tensor = transform(Image.fromarray(tile))

           
            for di in range(3):
                for dj in range(3):
                    target_i = di * grid_h_orig + i
                    target_j = dj * grid_w_orig + j

                    if (target_i + target_j) % 2 == 0:
                        generated_tiles[(target_i, target_j)] = tile_tensor.to(device)
                        original_tiles_list.append(tile_tensor)

   
    
    if len(original_tiles_list) > 0:
        original_tiles_tensor = torch.stack(original_tiles_list)  # [N, 3, H, W]
        orig_mean = original_tiles_tensor.mean(dim=[0, 2, 3])  # [3]
        orig_std = original_tiles_tensor.std(dim=[0, 2, 3])    # [3]
        print(f"Original color stats - Mean: {orig_mean.cpu().numpy()}, Std: {orig_std.cpu().numpy()}")
        
        orig_mean = orig_mean.to(device).view(3, 1, 1)  # [3, 1, 1]
        orig_std = orig_std.to(device).view(3, 1, 1)    # [3, 1, 1]
    else:
        orig_mean = None
        orig_std = None

    positions_to_generate = []
    for i in range(grid_h):
        for j in range(grid_w):
            if (i, j) not in generated_tiles:
                positions_to_generate.append((i, j))

    print(f"Need to generate {len(positions_to_generate)} new tiles")

    def get_neighbor_count(i, j, tiles_dict):
        neighbor_positions = [
            (i-1, j-1), (i-1, j), (i-1, j+1),
            (i, j-1),            (i, j+1),
            (i+1, j-1), (i+1, j), (i+1, j+1)
        ]
        return sum(1 for ni, nj in neighbor_positions if (ni, nj) in tiles_dict)

    pbar = tqdm(total=len(positions_to_generate), desc="Generating tiles")

    while positions_to_generate:
        positions_to_generate.sort(
            key=lambda pos: get_neighbor_count(pos[0], pos[1], generated_tiles),
            reverse=True
        )

        i, j = positions_to_generate.pop(0)

        neighbor_positions = [
            (i-1, j-1), (i-1, j), (i-1, j+1),
            (i, j-1),            (i, j+1),
            (i+1, j-1), (i+1, j), (i+1, j+1)
        ]

        neighbors = []
        neighbor_mask = []

        for ni, nj in neighbor_positions:
            if (ni, nj) in generated_tiles:
                neighbor_tile = generated_tiles[(ni, nj)].clone()
                neighbors.append(neighbor_tile)
                neighbor_mask.append(1.0)
            else:
                neighbors.append(torch.zeros(3, tile_size, tile_size, device=device))
                neighbor_mask.append(0.0)

        neighbors = torch.stack(neighbors)
        neighbor_mask = torch.tensor(neighbor_mask, dtype=torch.float32, device=device)

        with torch.no_grad():
            generated_tile = diffusion.sample(neighbors, neighbor_mask, tile_size=tile_size)
            generated_tile = generated_tile.squeeze(0) 
            
            if orig_mean is not None and orig_std is not None:
                gen_mean = generated_tile.mean(dim=[1, 2], keepdim=True) 
                gen_std = generated_tile.std(dim=[1, 2], keepdim=True)   
                
                
                generated_tile = (generated_tile - gen_mean) / (gen_std + 1e-8)
                
                
                generated_tile = generated_tile * orig_std + orig_mean
                
                
                generated_tile = torch.clamp(generated_tile, -1.0, 1.0)

        generated_tiles[(i, j)] = generated_tile.clone()

        pbar.update(1)
        pbar.set_postfix({
            'neighbors': f'{neighbor_mask.sum():.0f}/8',
            'pos': f'({i},{j})'
        })

    pbar.close()

    print("\nCompositing final image...")
    for (i, j), tile in tqdm(generated_tiles.items(), desc="Placing tiles"):
        y_start = i * (tile_size - overlap)
        x_start = j * (tile_size - overlap)

        weight = create_blend_weight(tile_size, overlap).to(device)

        canvas[:, y_start:y_start+tile_size, x_start:x_start+tile_size] += tile * weight
        weight_map[:, y_start:y_start+tile_size, x_start:x_start+tile_size] += weight

    canvas = canvas / (weight_map + 1e-8)

    canvas = canvas.cpu()
    canvas = canvas * 0.5 + 0.5  
    canvas = torch.clamp(canvas, 0, 1)
    
    canvas = canvas.numpy().transpose(1, 2, 0)
    canvas = (canvas * 255).astype(np.uint8)
    canvas = canvas[:orig_h*3, :orig_w*3]

    return Image.fromarray(canvas)



def create_blend_weight(tile_size, overlap):
    
    weight = torch.ones(1, tile_size, tile_size)

    if overlap > 0:
        
        for i in range(tile_size):
            for j in range(tile_size):
                dist_from_edge = min(i, j, tile_size-1-i, tile_size-1-j)
                if dist_from_edge < overlap:
                    
                    alpha = (dist_from_edge + 1) / (overlap + 1)
                    alpha = alpha ** 2  
                    weight[0, i, j] = alpha

    return weight


def visualize_predictions(model, diffusion, dataset, num_samples=4, split_name="Train"):
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for idx in range(num_samples):
            sample_idx = random.randint(0, len(dataset) - 1)
            center_tile, neighbors, neighbor_mask = dataset[sample_idx]
            
            center_tile = center_tile.unsqueeze(0).to(device)
            neighbors = neighbors.unsqueeze(0).to(device)
            neighbor_mask = neighbor_mask.unsqueeze(0).to(device)
            
            t = torch.randint(50, 80, (1,), device=device)
            noisy_tile, actual_noise = diffusion.forward_diffusion(center_tile, t)
            
            predicted_noise = model(noisy_tile, t, neighbors, neighbor_mask)
            
            alpha_cumprod_t = diffusion.alphas_cumprod[t.item()]
            pred_x0 = (noisy_tile - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            def to_display(tensor):
                img = tensor.squeeze(0).cpu()
                img = (img + 1.0) / 2.0
                img = torch.clamp(img, 0, 1)
                return img.permute(1, 2, 0).numpy()
            
            axes[idx, 0].imshow(to_display(center_tile))
            axes[idx, 0].set_title(f'Original (t={t.item()}, valid={neighbor_mask.sum().item():.0f}/8)')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(to_display(noisy_tile))
            axes[idx, 1].set_title('Noisy Input')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(to_display(pred_x0))
            axes[idx, 2].set_title('Denoised (Predicted)')
            axes[idx, 2].axis('off')
            
            
            noise_error = torch.abs(predicted_noise - actual_noise).mean(dim=1, keepdim=True)
            noise_error_display = noise_error.squeeze().cpu().numpy()
            im = axes[idx, 3].imshow(noise_error_display, cmap='hot')
            axes[idx, 3].set_title('Noise Error (MSE)')
            axes[idx, 3].axis('off')
            plt.colorbar(im, ax=axes[idx, 3], fraction=0.046)
    
    plt.suptitle(f'{split_name} Set Predictions', fontsize=16, y=1.0)
    plt.tight_layout()
    return fig


def main():
    # Configuration
    INPUT_IMAGE = "texture.png"
    OUTPUT_IMAGE = "output_texture_3x32.jpg"
    TILE_SIZE = 128
    NUM_TRAINING_SAMPLES = 1000
    EPOCHS = 0
    PATIENCE = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0002
    MISSING_NEIGHBOR_PROB = 0.3
    FORCE_MISSING_PATTERNS = True
    MODEL_PATH = 'texture_diffusion_ddpm2.pth'
    CONFIG_PATH = 'diffusion_config2.pth'

    # Diffusion parameters
    TIMESTEPS = 100
    BETA_START = 0.0001
    BETA_END = 0.02

    if not os.path.exists(INPUT_IMAGE):
        return
    
    
    if EPOCHS > 0:
        model, diffusion, diffusion_config, train_dataset, val_dataset = train_diffusion(
            INPUT_IMAGE,
            tile_size=TILE_SIZE,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            num_samples=NUM_TRAINING_SAMPLES,
            patience=PATIENCE,
            missing_neighbor_prob=MISSING_NEIGHBOR_PROB,
            force_missing_patterns=FORCE_MISSING_PATTERNS,
            timesteps=TIMESTEPS,
            beta_start=BETA_START,
            beta_end=BETA_END
        )
        torch.save(model.state_dict(), MODEL_PATH)
        torch.save(diffusion_config, CONFIG_PATH)
    else:

       
        model = UNet(in_channels=3, out_channels=3, neighbor_channels=24).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        if os.path.exists(CONFIG_PATH):
            diffusion_config = torch.load(CONFIG_PATH, map_location=device)
        else:
            diffusion_config = {
                'timesteps': TIMESTEPS,
                'beta_start': BETA_START,
                'beta_end': BETA_END
            }
        diffusion = DDPMDiffusion(
            model,
            timesteps=diffusion_config['timesteps'],
            beta_start=diffusion_config['beta_start'],
            beta_end=diffusion_config['beta_end']
        )
        total_samples = NUM_TRAINING_SAMPLES
        train_samples = int(total_samples * 0.85)
        val_samples = total_samples - train_samples
        
        train_dataset = TextureTileDataset(
            INPUT_IMAGE, tile_size=TILE_SIZE, num_samples=train_samples,
            missing_neighbor_prob=MISSING_NEIGHBOR_PROB,
            force_missing_patterns=FORCE_MISSING_PATTERNS
        )
        val_dataset = TextureTileDataset(
            INPUT_IMAGE, tile_size=TILE_SIZE, num_samples=val_samples,
            missing_neighbor_prob=MISSING_NEIGHBOR_PROB,
            force_missing_patterns=False
        )
    
    train_fig = visualize_predictions(model, diffusion, train_dataset, num_samples=4, split_name="Training")
    train_fig.savefig('train_predictions.png', dpi=150, bbox_inches='tight')
    plt.close(train_fig)
    
    val_fig = visualize_predictions(model, diffusion, val_dataset, num_samples=4, split_name="Validation")
    val_fig.savefig('val_predictions.png', dpi=150, bbox_inches='tight')
    plt.close(val_fig)
    
    output_texture = synthesize_3x3_texture_checkerboard(
        model, diffusion, INPUT_IMAGE,
        tile_size=TILE_SIZE,
        overlap=8
    )

    output_texture.save(OUTPUT_IMAGE)
    

if __name__ == "__main__":
    main()
