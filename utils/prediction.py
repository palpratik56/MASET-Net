def show_test_predictions(test_loader, model, device, num_images=5):
    
    torch.cuda.empty_cache()  # Clear GPU cache before starting
    model.eval()  # Set model to evaluation mode
    fig, axs = plt.subplots(3, num_images, figsize=(10, 8))
    
    # Get a batch of images that's large enough
    images, masks = next(iter(test_loader))
    batch_size = images.size(0)
    
    # Generate unique random indices
    selected_indices = random.sample(range(batch_size), num_images)
    
    with torch.no_grad():  # No need for gradients in evaluation
        for i, idx in enumerate(selected_indices):
            # Use the randomly selected index
            image = images[idx].to(device)
            mask = masks[idx].to(device)
            
            # Get the predicted mask from the model
            output,_ = model(image.unsqueeze(0))  # Add batch dimension (1, C, H, W)
            predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy()  # Apply sigmoid and remove batch dim
            
            # Convert image and mask tensors to numpy arrays for visualization
            image_np = image.cpu().permute(1, 2, 0).numpy()  # Convert CHW to HWC
            mask_np = mask.cpu().squeeze().numpy()  # Convert mask to numpy
            
            # Plot the original image
            axs[0, i].imshow(image_np)
            axs[0, i].set_title(f"Image {i+1}")
            axs[0, i].axis('off')
            
            # Plot the ground truth mask
            axs[1, i].imshow(mask_np, cmap='gray')
            axs[1, i].set_title(f"Ground Truth {i+1}")
            axs[1, i].axis('off')
            
            # Plot the predicted mask
            axs[2, i].imshow(predicted_mask, cmap='gray')
            axs[2, i].set_title(f"Predicted Mask {i+1}")
            axs[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save the figure
    fig.savefig('ISIC_predictions_RA.png', dpi=600)
    # Clean up
    plt.close(fig)
