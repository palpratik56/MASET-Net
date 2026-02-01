import matplotlib as mpl

#extract the metrices from the results dictionary
# training_metrics = results["B7_Full_MASET"]['train']
# validation_metrics = results["B7_Full_MASET"]['validation']

training_metrics = metrics['train']
validation_metrics = metrics['validation']

def plot_training_metrics(training_metrics, validation_metrics, save_path=None):
    # Set style for better visualization
    plt.style.use('default')

    mpl.rcParams.update({
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'grid.color': 'black',
        'grid.linestyle': '--',
        'grid.alpha': 0.4,
        'axes.edgecolor': 'black',
    })

    
    # Create figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 12), sharex=True)
    # fig.suptitle('CVC Performance Metrics', fontsize=18, y=0.95)
    
    # Flatten axs for easier iteration
    axs = axs.flatten()
    
    # Metrics to plot (excluding dice since it's similar to accuracy)
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'dice', 'iou']
    titles = ['Loss', 'Accuracy', 'Precision', 'Recall', 'Dice', 'IoU']
    
    # Plot each metric
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        # Plot training metric
        axs[idx].plot(training_metrics[metric], 
                     label=f'Training {title}', 
                     color='green', 
                     linewidth=2)
        
        # Plot validation metric
        axs[idx].plot(validation_metrics[metric], 
                     label=f'Validation {title}', 
                     color='orange', 
                     linewidth=2)
        
        # Customize subplot
        axs[idx].set_title(f'{title}', pad=10)
        axs[idx].set_xlabel('Epoch')
        # axs[idx].set_ylabel(title)
        axs[idx].set_facecolor('white')
        axs[idx].grid(True, linestyle='--', alpha=0.7)
        axs[idx].legend(loc='best')
        
        # Add horizontal and vertical grid
        axs[idx].grid(True, which='both', linestyle='--', alpha=0.6)
        
        # Set y-axis limits for metrics other than loss
        if metric != 'loss':
            axs[idx].set_ylim([0, 1.1])
    

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save plot if path is provided
    if save_path:
         plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
    plt.show()

# Create a separate figure for Dice Score
def plot_dice_score(training_metrics, validation_metrics, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(training_metrics['dice'], label='Training Dice', color='violet', linewidth=2)
    plt.plot(validation_metrics['dice'], label='Validation Dice', color='orange', linewidth=2)
    
    plt.title('Dice Score w.r.t AdamW Optimizer', pad=20, fontsize=14)
    plt.xlabel('Epoch')
    # plt.ylabel('Dice Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.ylim([0, 1.1])
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    plt.show()
