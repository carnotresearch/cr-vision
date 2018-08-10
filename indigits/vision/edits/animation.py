'''
Simple animations
'''
import os
import click
import imageio
from skimage import transform, util
from skimage import filters, color
from indigits import vision as iv


def stretch_and_shrink(image, num_steps=40):
    '''
    Stretches an image by some mount and shrinks it back over a one second animation
    '''
    height, width = image.shape[:2]
    extra = int (width / 10.0)    
    step = int(extra * 1.0 / num_steps)
    # make sure that step size is even
    if step % 2 == 1: 
        step  = step + 1
    images = [image]
    image = util.img_as_float(image)
    print('Image size: {}x{}'.format(width, height))
    print('Number of seams to be removed: {}, in each iteration: {}'.format(num_steps * step, step))

    for i in range(num_steps):
        print('Iteration: {}'.format(i+1))
        #print('Converting to gray scale.')
        gray_image = color.rgb2gray(image)
        #print('Computing sobel energy.')
        energy = filters.sobel(gray_image)
        #print('Performing seam carving')
        #print(image.shape)
        image = transform.seam_carve(image, energy, 'vertical', step)
        #print('Converting to 8-bit format')
        image_u8 = util.img_as_ubyte(image)
        #print('Adding side bars in black')
        cur_width = image_u8.shape[1]
        # add pillar box appropriately
        image_u8 = iv.add_pillar_box_pattern(image_u8, int((width - cur_width)/2))
        print(image_u8.shape)
        #print(image_u8.shape)
        images.append(image_u8)
    images_reversed = images[::-1]
    images.extend(images_reversed)
    return images


@click.command()
@click.argument('input_image_path')
@click.argument('output_movie_path')
def stretch_and_shrink_app(input_image_path, output_movie_path):
    '''Wrapper application for stretch and shrink effect'''
    print('Reading {}'.format(input_image_path))
    image = imageio.imread(input_image_path)
    images = stretch_and_shrink(image)
    height, width = images[0].shape[:2]
    print('Writing {} images to {}'.format(len(images), output_movie_path))
    if os.path.isdir(output_movie_path):
        # we will write images one by one
        for (i, image) in enumerate(images):
            imageio.imsave(os.path.join(
                output_movie_path, '{}.jpg'.format(i)), image)
    else:
        # we write the whole animation
        # writer = imageio.get_writer(output_movie_path, fps=5)
        # for image in images:
        #     writer.append_data(image)
        # writer.close()
        writer = iv.io.VideoWriter(output_movie_path, fps=10, frame_size=(width, height))
        for image in images:
            image = iv.rgb_to_bgr(image)
            writer.write(image)
        writer.stop()
    return

if __name__ == '__main__':
    stretch_and_shrink_app()
