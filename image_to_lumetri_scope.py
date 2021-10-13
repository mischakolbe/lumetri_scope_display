"""Module to convert an image to show up in a lumetri scope.

Done out of curiosity and for fun. This code isn't clean or efficient, but it
should be documented enough to get you started if you want to tinker with it <3
"""
import colorsys
import os
import pathlib
import numpy as np
import random
import time
from functools import partial
from multiprocessing import Pool
import math

from PIL import Image, ImageOps

import dither


def image_to_waveform_rgb(
    input_path,
    output_path,
    output_width=1920,
    output_height=1080,
    style=None,
    dither_input=True,
    color_threshold=160,
    **kwargs
):
    """Convert an image to show up in a Waveform RGB display.

    Args:
        input_path (str): Image to process.
        output_path (str): Path where the output image should be saved to.
        output_width (int): Horizontal resolution of output image. This
            determines how much "storage" you have to save the converted pixels.
        output_height (int): Vertical resolution of output image. This
            determines how much "storage" you have to save the converted pixels.
        style (str or None): Apply different "styles" to the output image. This
            takes advantage of the fact that the vertical order of pixels
            doesn't matter and can therefore be rearranged. Options are:
            None, "random", "sorted", "rgb_sorted".
        dither_input (bool): Whether the input image should be dithered.
        color_threshold (int): The color intensity at which a pixel is
            considered to contain that color. For example a pixel with rgb of
            (20, 150, 180) will become (0, 0, 255) with a threshold of 160.

    Returns:
        str: Path of the converted image.

    Examples:
        ::

            # Process a single image to Waveform
            output_image = image_to_waveform_rgb(
                input_path=r"some\input\image.jpg",
                output_path=r"your\output\waveform_image.png",
                output_width=1920,
                output_height=1080,
                style="rgb_sorted",
                dither_input=True,
                color_threshold=160,
            )
    """
    input_image = Image.open(input_path)

    # Fit the given image into the output resolution.
    image_to_process = ImageOps.contain(
        input_image, (output_width, output_height)
    )
    input_image.close()
    del input_image

    # Dither the image before processing it.
    if dither_input:
        image_to_process = dither.convert(image_to_process)

    # Determine offsets to center the resized image.
    resized_width = image_to_process.width
    resized_height = image_to_process.height
    horizontal_offset = int((output_width - resized_width) / 2)
    vertical_offset = int((output_height - resized_height) / 2)

    # Create an empty output image to fill in the loop.
    output_image = Image.new(
        mode="RGB", size=(output_width, output_height), color=0
    )

    for pixel_horizontal in range(resized_width):
        # Creating a column list makes it easier to apply styles later on.
        current_column = []

        offset_horizontal_index = pixel_horizontal + horizontal_offset
        for pixel_vertical in range(resized_height):
            r, g, b = image_to_process.getpixel((pixel_horizontal, pixel_vertical))
            intensity = int(255.0 * (resized_height - pixel_vertical) / resized_height)
            red = intensity if r > color_threshold else 0
            green = intensity if g > color_threshold else 0
            blue = intensity if b > color_threshold else 0
            current_column.append((red, green, blue))

        # The vertical position of pixels doesn't matter (luminance determines
        # vertical position in lumetri scope). Therefore we can apply different
        # "styles" to the image, by reordering pixels vertically.
        if style and style == "random":
            # Shuffle the pixels randomly
            random.shuffle(current_column)
        elif style and style == "sorted":
            # Sort the pixels as-is in ascending RGB brightness.
            current_column.sort()
        elif style and style == "rgb_sorted":
            # Since the RGB values of a pixel are treated individually in the
            # Waveform display anyways they don't need to remain together.
            # This sorting style takes advantage of this and sorts the RGB
            # channels individually.
            sorted_column = [
                sorted(list(column)) for column in zip(*current_column)
            ]
            current_column = [
                (r, g, b) for r, g, b in zip(*sorted_column)
            ]

        # Store the pixels into the output image.
        for offset_vertical_index, color in enumerate(current_column, vertical_offset):
            output_image.putpixel(
                (offset_horizontal_index, offset_vertical_index), color
            )

    image_to_process.close()
    del image_to_process

    # Store the image in a lossless format.
    output_image.save(os.path.splitext(output_path)[0] + ".png", "PNG")
    output_image.close()
    del output_image

    return output_path


def image_to_vectorscope_hls(
    input_path,
    output_path,
    output_width=1920,
    output_height=1080,
    style="sorted",
    scale_to_fit=True,
    output_resolution=512,
    bit_depth=9,
    outline_with_remaining_pixels=True,
    maintain_chunk_positions=True,
    chunk_fill_color=None,
    **kwargs
):
    """Convert an image to show up in a Vectorscope HLS display.

    Args:
        input_path (str): Image to process.
        output_path (str): Path where the output image should be saved to.
        output_width (int): Horizontal resolution of output image. This
            determines how much "storage" you have to save the converted pixels.
        output_height (int): Vertical resolution of output image. This
            determines how much "storage" you have to save the converted pixels.
        style (str or None): Apply different "styles" to the output image. This
            takes advantage of the fact that the order of pixels doesn't matter
            and can therefore be rearranged. Options are:
            None, "random", "sorted".
        scale_to_fit (bool): Whether the input image should be scaled to
            fill the output resolution.
        output_resolution (int): Expected resolution of the Vectorscope. Since
            the Vectorscope is round only one dimension is needed (diameter).
        bit_depth (int): Number of possible shades of grey.
        outline_with_remaining_pixels (bool): Whether pixels that aren't used
            to represent the actual image should be used to create an outline
            around the Vectorscope border.
        maintain_chunk_positions (bool): Whether the chunks of pixels in the
            output image that make up the grey-tone of one dot in the
            Vectorscope should remain in the same location, even if they aren't
            filled.
        chunk_fill_color (None or tuple(int)): The color that should be used
            for the empty pixels in a chunk.

    Returns:
        str: Path of the converted image.

    Examples:
        ::

            # Process a single image to Vectorscope
            output_image = image_to_vectorscope_hls(
                input_path=r"some\input\image.jpg",
                output_path=r"your\output\vectorscope_image.png",
                output_width=1920,
                output_height=1080,
                style="sorted",
                scale_to_fit=True,
                output_resolution=512,
                bit_depth=8,
                outline_with_remaining_pixels=True,
                maintain_chunk_positions=True,
                chunk_fill_color=None,
            )
    """
    # Resize the image to fit into the output image.
    input_image = Image.open(input_path)
    input_width = float(input_image.width)
    input_height = float(input_image.height)

    # Stretch image to fit the output resolution (either min or max dimension)
    if scale_to_fit:
        new_ratio = max(
            output_resolution/input_width, output_resolution/input_height
        )
    else:
        new_ratio = min(
            output_resolution/input_width, output_resolution/input_height
        )

    new_size = (int(input_width * new_ratio), int(input_height * new_ratio))
    resized_image = input_image.resize(new_size)

    input_image.close()
    del input_image

    # Find recommended max bit depth and warn user if pixels might overflow.
    total_output_pixels = output_width * output_height
    # The pixels that aren't inside the circular Vectorscope are ignored.
    pixels_per_bit_depth = output_resolution ** 2 * math.pi / 4
    recommended_max_bit_depth = math.floor(total_output_pixels/pixels_per_bit_depth)
    # Adjust the bit_depth value to get useful dividers.
    adjusted_bit_depth = bit_depth + 1
    if adjusted_bit_depth > recommended_max_bit_depth:
        message = (
            "The recommended bit depth for your output resolutions is {}. "
            "Since you went for a higher bit depth ({}) your values might "
            "overflow, which results in an incomplete picture."
        )
        print(message.format(recommended_max_bit_depth - 1, bit_depth))

    # Center image into output resolution
    centered_image = Image.new(
        mode="RGB", size=(output_resolution, output_resolution), color=0
    )
    centered_image.paste(
        resized_image,
        (
            int((output_resolution - resized_image.width) / 2),
            int((output_resolution - resized_image.height) / 2),
        ),
    )
    resized_image.close()
    del resized_image

    # Convert image into greyscale
    luminance_image = centered_image.convert("L")
    luminance_image_array = np.array(luminance_image)

    centered_image.close()
    del centered_image

    # Initialize output array
    output_image = Image.new(
        mode="RGB", size=(output_width, output_height), color=0
    )

    half_output_resolution = output_resolution / 2.0
    center_point = np.array([half_output_resolution, half_output_resolution])
    image_array = np.array(output_image)

    # Reshape image array into a single array for easier processing & sorting.
    flat_image_array = np.reshape(
        image_array, (output_image.height * output_image.width, 3)
    )

    # Initialize color array to pick from, for the outline.
    outline_steps = 720
    outline_colors = []
    for angle in range(outline_steps):
        # The outline is defined by the fully saturated color of every hue.
        hue = colorsys.hls_to_rgb(h=angle/float(outline_steps), l=0.5, s=0.99)
        outline_colors.append(tuple(int(color*255) for color in hue))
    current_outline_color_index = 0

    # Some constants that are used often during the loop.
    angle_offset = math.pi / 2.0
    bit_depth_divisor = 256.0 / adjusted_bit_depth

    current_output_array_index = 0
    num_output_pixels = flat_image_array.shape[0]
    for pixel_vertical, column in enumerate(luminance_image_array):
        for pixel_horizontal, value in enumerate(column):

            pixel_pos = np.array([pixel_horizontal, pixel_vertical])
            vector_origin_pixel = pixel_pos - center_point

            # Skip over pixels that are outside vector scope "display area"
            # In other words: Outside the circle inside the square target image.
            magnitude = np.linalg.norm(vector_origin_pixel)
            if magnitude > half_output_resolution:
                continue

            # Calculate the angle and magnitude to position the dot in the
            # polar coordinate system of the Vectorscope.
            normalized_magnitude = magnitude / 256.0
            angle = np.arctan2(vector_origin_pixel[1], vector_origin_pixel[0])
            normalized_angle = -(angle + angle_offset) / math.tau
            # Convert the hue (=angle) and saturation (=magnitude) to RGB.
            rgb = colorsys.hls_to_rgb(
                h=normalized_angle, l=0.5, s=normalized_magnitude
            )
            color_for_positioning = tuple(int(c*255) for c in rgb)

            # Create multiple pixels to fake greyscale values
            repeats_for_value = int(value / bit_depth_divisor)
            for repeat_index in range(adjusted_bit_depth):
                if repeat_index >= repeats_for_value:
                    # Skip if there are no more repeats to do and data chunks
                    # for output pixels can vary in size.
                    if not maintain_chunk_positions:
                        continue

                    # Use a static color to fill chunk, if it should remain "clean".
                    if chunk_fill_color:
                        color_to_add = chunk_fill_color

                    # Use outline color to fill chunk.
                    else:
                        color_to_add = outline_colors[current_outline_color_index]
                        current_outline_color_index = (
                            (current_outline_color_index + 1) % outline_steps
                        )

                # Use the calculated color to place a pixel in the Vectorscope.
                else:
                    color_to_add = color_for_positioning

                # Break out of loop, if max index was reached.
                try:
                    flat_image_array[current_output_array_index] = color_to_add
                except IndexError:
                    break

                current_output_array_index = current_output_array_index + 1

    luminance_image.close()
    del luminance_image

    # Fill remaining pixels with colors that will form an outline in the Vectorscope.
    remaining_output_pixels = num_output_pixels - current_output_array_index
    if outline_with_remaining_pixels and remaining_output_pixels:
        repeated_outline_array = np.resize(outline_colors, (remaining_output_pixels, 3))
        flat_image_array[current_output_array_index:] = repeated_outline_array

    if style:
        if style == "random":
            # Randomize pixels
            np.random.shuffle(flat_image_array)
        elif style == "sorted":
            # Sort pixels based on sort order (r=0, g=1, b=2)
            sort_order = [0, 1, 2]
            first_iteration = True
            for sort_order_index in reversed(sort_order):
                if first_iteration:
                    # First sort doesn't need to be stable.
                    arg_sorted = flat_image_array[:,sort_order_index].argsort()
                    flat_image_array = flat_image_array[arg_sorted]
                else:
                    merge_sorted = flat_image_array[:,sort_order_index].argsort(
                        kind="mergesort"
                    )
                    flat_image_array = flat_image_array[merge_sorted]
                first_iteration=False

    # Reshape image array into the output image dimensions.
    image_array = np.reshape(
        flat_image_array, (output_image.height, output_image.width, 3)
    )

    # Store the image in a lossless format.
    output_image = Image.fromarray(image_array.astype("uint8"))
    output_image.save(os.path.splitext(output_path)[0] + ".png", "PNG")
    output_image.close()
    del output_image

    return output_path


def process_image(
    image,
    conversion_function,
    input_folder,
    output_folder,
    output_width=1920,
    output_height=1080,
    replace_existing=False,
    **kwargs,
):
    """Process a single image with the given conversion_function.

    Args:
        image (str): Name of the image to process.
        conversion_function (func): The function to use for the conversion.
            Either "image_to_waveform_rgb" or "image_to_vectorscope_hls".
        input_folder (str): Folder with images to process. Note: The script
            will attempt to process ALL files inside this folder.
        output_folder (str): Folder where the output images should be saved to.
        output_width (int): Horizontal resolution of output image. This
            determines how much "storage" you have to save the converted pixels.
        output_height (int): Vertical resolution of output image. This
            determines how much "storage" you have to save the converted pixels.
        replace_existing (bool): Whether to overwrite existing files.
        kwargs (dict): Keyword arguments to be passed on to the given
            conversion_function.

    Returns:
        str: Path of the converted image.
    """
    output_path = os.path.join(output_folder, image)
    if os.path.isfile(output_path) and not replace_existing:
        print("Skipping {}: Already exists.".format(output_path))
        return output_path

    print("Processing:", image)
    input_path = os.path.join(input_folder, image)
    output_image = conversion_function(
        input_path=input_path,
        output_path=output_path,
        output_width=output_width,
        output_height=output_height,
        **kwargs,
    )

    return output_image


def process_folder(
    conversion_function,
    input_folder,
    output_folder,
    output_width=1920,
    output_height=1080,
    replace_existing=False,
    threads=8,
    **kwargs,
):
    """Process all images in a folder.

    Args:
        conversion_function (func): The function to use for the conversion.
            Either "image_to_waveform_rgb" or "image_to_vectorscope_hls".
        input_folder (str): Folder with images to process. Note: The script
            will attempt to process ALL files inside this folder.
        output_folder (str): Folder where the output images should be saved to.
        output_width (int): Horizontal resolution of output image. This
            determines how much "storage" you have to save the converted pixels.
        output_height (int): Vertical resolution of output image. This
            determines how much "storage" you have to save the converted pixels.
        replace_existing (bool): Whether to overwrite existing files.
        threads (int): How many threads should be used while processing.
        kwargs (dict): Keyword arguments to be passed on to the given
            conversion_function.

    Examples:
        ::

            # Process an entire folder to Waveform
            process_folder(
                conversion_function=image_to_waveform_rgb,
                input_folder=r"some\input\folder",
                output_folder=r"your\output\folder",
                threads=8,
                replace_existing=False,
                output_width=1920,
                output_height=1080,
                style="rgb_sorted",
                dither_input=True,
                color_threshold=160,
            )

            # Process an entire folder to Vectorscope
            process_folder(
                conversion_function=image_to_vectorscope_hls,
                input_folder=r"some\input\folder",
                output_folder=r"your\output\folder",
                threads=8,
                replace_existing=False,
                output_width=1920,
                output_height=1080,
                style="sorted",
                scale_to_fit=True,
                output_resolution=512,
                bit_depth=8,
                outline_with_remaining_pixels=True,
                maintain_chunk_positions=True,
                chunk_fill_color=None,
            )
    """
    start_time = time.time()

    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    _, _, filenames = list(os.walk(input_folder))[0]

    with Pool(threads) as pool:
        pool.map(
            partial(
                process_image,
                conversion_function=conversion_function,
                input_folder=input_folder,
                output_folder=output_folder,
                output_width=output_width,
                output_height=output_height,
                replace_existing=replace_existing,
                **kwargs,
            ),
            filenames,
        )

    message = "Took: {} seconds to process {} images with {} threads."
    print(message.format(time.time() - start_time, len(filenames), threads))
