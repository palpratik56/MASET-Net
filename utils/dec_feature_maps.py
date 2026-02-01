def visualize_feature_maps(feature_maps, ground_truths):
    num_maps = len(feature_maps)
    fig, axs = plt.subplots(1, num_maps + 1, figsize=(14, 6))
    
    # Randomly select an image index from the batch
    batch_size = feature_maps[list(feature_maps.keys())[0]].shape[0]
    random_idx = torch.randint(0, batch_size, (1,)).item()
    
    for i, (name, feature_map) in enumerate(feature_maps.items()):
        # Feature map visualization (use first channel and normalize)
        feature_map_np = feature_map[random_idx, 0].detach().cpu().numpy()
        feature_map_np = (feature_map_np - feature_map_np.min()) / (feature_map_np.max() - feature_map_np.min())
        
        # Feature map subplot  
        axs[i].imshow(feature_map_np, cmap='viridis')
        if i == 4:
            axs[i].set_title('Output')
        else:
            axs[i].set_title(f'Stage: {i+1}')
        axs[i].axis('off')
    
    # Ground truth visualization at the end
    # Using the same random_idx for consistency
    ground_truth_np = ground_truths[random_idx].detach().cpu().numpy()
    ground_truth_np = ground_truth_np.squeeze(0)  # Squeeze the first dimension (channel)
    
    axs[-1].imshow(ground_truth_np, cmap='viridis')
    axs[-1].set_title('Ground Truth')
    axs[-1].axis('off')
    
    plt.tight_layout()
    
    # Add the random index to the save path to track different visualizations
    plt.savefig('ISIC_feature_maps_RA.png', dpi=600)
    plt.show()
