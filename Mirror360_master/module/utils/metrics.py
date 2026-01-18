import torch
def calculate_and_print_metrics(original_chunk_np, processed_tensor, device, metrics_calculators):
    """Calculates and prints PSNR, SSIM, LPIPS, and VMAF for a given chunk."""
    print("[cyan]Calculating quality metrics...[/cyan]")

    # Unpack metric calculators
    psnr_metric = metrics_calculators['psnr']
    ssim_metric = metrics_calculators['ssim']
    lpips_metric = metrics_calculators['lpips']
    ms_ssim_metric = metrics_calculators['ms_ssim']
    # Prepare original tensor: convert from numpy, normalize, and send to device
    original_tensor = torch.from_numpy(original_chunk_np).float().to(device)

    # Ensure tensors have the same length (for the last chunk which might be shorter)
    min_len = min(original_tensor.shape[0], processed_tensor.shape[0])
    original_tensor = original_tensor[:min_len]
    processed_tensor_comp = processed_tensor[:min_len].detach()

    # --- PSNR and SSIM ---
    psnr_val = psnr_metric(processed_tensor_comp, original_tensor)
    ssim_val = ssim_metric(processed_tensor_comp, original_tensor)
    ms_ssim_val = ms_ssim_metric(processed_tensor_comp, original_tensor)
    # --- LPIPS ---
    # LPIPS expects input range [-1, 1], so we rescale
    processed_lpips = processed_tensor_comp * 2 - 1
    original_lpips = original_tensor * 2 - 1
    lpips_val = lpips_metric(processed_lpips, original_lpips).mean()

    print(f"[bold magenta]Metrics for this chunk:[/bold magenta]")
    print(f"  - PSNR: {psnr_val.item():.4f}")
    print(f"  - SSIM: {ssim_val.item():.4f}")
    print(f"  - LPIPS: {lpips_val.item():.4f}")
    print(f"  - MS-SSIM: {ms_ssim_val.item():.4f}")

