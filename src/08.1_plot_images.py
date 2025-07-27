import matplotlib.pyplot as plt
import imageio

def plot_images(image_paths, id):
    # Create a figure and axes
    fig, axs = plt.subplots(1, len(image_paths), figsize=(15, 5))

    for i, path in enumerate(image_paths):
        # Read the image
        img = imageio.imread(path)
        
        # Display the image on the corresponding axis
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')  # Turn off axis numbers and ticks

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plot
    plt.savefig(f'./checkpoints/results/images_plot_{id}.png')
    plt.close()

# Example usage
for i in range(100):
    image_paths = [f'data/blurred/{i:07}_blur.png', f'data/deblurred/{i:07}_deblur.png', f'data/dewarped/{i:07}_dewarp.png']
    plot_images(image_paths, i)