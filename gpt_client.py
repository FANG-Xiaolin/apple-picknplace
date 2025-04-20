#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gpt_client.py
# Author : Xiaolin Fang
# Email  : fxlfang@gmail.com
# Date   : 04/17/2025
#
# Distributed under terms of the MIT license.

"""

"""
import base64
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from utils import plot_images

OPENAI_ORGANIZATION_KEY = ""
OPENAI_PROJECT_KEY = None
OPENAI_API_KEY = os.environ['OPENAI_API_KEY'] # TODO: from config


def extract_tag_content(content: str, tag: str):
    """
    Extract the content of a tag from a string.
    Args:
        content:    str. The content string
        tag:        str. The tag to extract

    Returns:
        str. The content of the tag.
    """
    start = content.find(f"<{tag}>")
    end = content.find(f"</{tag}>")
    if start == -1 or end == -1:
        return None
    return content[start + len(tag) + 2 : end]


class GPTClient:
    def __init__(self, verbose_level=0):
        self.client = OpenAI(
            organization=OPENAI_ORGANIZATION_KEY,
            project=OPENAI_PROJECT_KEY,
            api_key=OPENAI_API_KEY,
        )
        self.verbose_level = verbose_level

    def obtain_GPT_naming(self, image_sequence: list[np.ndarray]):
        """
        Prompt GPT to obtain the name of an object.
        Args:
            sequence: rgb image sequence
        Returns:

        """
        instruction = f'I have provided a few images. Please return the name for the object to each image.'
        instruction += "Your output should be multiple lines, each line corresponding to an image of the format <output>{object_name}</output>. Do not return anything else."

        b64_images = []  #  base64 encoded images
        for img in image_sequence:
            encoded_file = cv2.imencode(".jpg", img[..., ::-1])
            encoded_image = base64.b64encode(encoded_file[1]).decode("utf-8")
            b64_images.append(encoded_image)

        messages = [
            {"role": "user", "content": [
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{im_i}"}} for im_i in b64_images],
                {"type": "text", "text": instruction},
            ],
        }]

        gpt_return_result = self.client.chat.completions.create(model="gpt-4o", messages=messages)
        if self.verbose_level > 0:
            print('='*10, 'GPT Returned Result', '='*10)
            print(gpt_return_result.choices[0].message.content)
            print('='*45)
        return messages, gpt_return_result

    def obtain_GPT_naming_auto(self, image_sequence: list[np.ndarray], max_retries=3):
        for i in range(max_retries):
            try:
                messages, result = self.obtain_GPT_naming(image_sequence)
                output_lines = result.choices[0].message.content.split('\n')
                object_names = [extract_tag_content(output_line, "output") for output_line in output_lines]
                if len(object_names) != len(image_sequence):
                    raise RuntimeError(f'GPT Result Parsing Error. Received {len(object_names)} results for {len(image_sequence)} queries.')
                return object_names
            except Exception as e:
                print(f"Error in GPT call. Retrying. Error: {e}")
        raise RuntimeError("GPT call failed.")

    def obtain_GPT_mostlikelyobject(self, image_sequence: list[np.ndarray], text_desc: str, allow_none=True):
        """
        Prompt GPT to select the image that matches the text description the most. Return 0 if None of them appears to be a sensible choice.
        Args:
            sequence: rgb image sequence
            text_desc: a target description
        Returns:
            image_id: index (1-based) of the best match image. 0 if None.
        """
        instruction = "I have provided you a sequence of images.\n"
        instruction += f"Please select the ID of the image that is most similar to the description '{text_desc}'.\n"
        instruction += "Put your answer in the format of <output>{ID}</output>\n"
        instruction += "Let's think step by step following the pattern:\n"
        instruction += "image_1_mask = '...' # Describe the mask in the first image\n"
        instruction += "image_2_mask = '...' # Describe the mask in the second image\n"
        instruction += "...\n"
        instruction += "output: <output>{ID}</output>  # return the image ID that matches the text description in the last line.\n"
        if allow_none:
            instruction += "If None of them matches the text description, return ID 0.\n"
        instruction += "Do not include anything else in the last line other than <output>{ID}</output>"

        b64_images = []  #  base64 encoded images
        for img in image_sequence:
            encoded_file = cv2.imencode(".jpg", img[..., ::-1])
            encoded_image = base64.b64encode(encoded_file[1]).decode("utf-8")
            b64_images.append(encoded_image)

        messages = [
            {"role": "user", "content": [
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{im_i}"}} for im_i in b64_images],
                {"type": "text", "text": instruction},
            ],
        }]

        gpt_return_result = self.client.chat.completions.create(model="gpt-4o", messages=messages)
        if self.verbose_level > 0:
            print('='*10, 'GPT Returned Result', '='*10)
            print(gpt_return_result.choices[0].message.content)
            print('='*45)
        return messages, gpt_return_result

    def obtain_GPT_mostlikelyobject_auto(self, image_sequence: list[np.ndarray], text_desc: str, max_retries=3, allow_none=True):
        min_valid_id = 0 if allow_none else 1
        for i in range(max_retries):
            try:
                messages, result = self.obtain_GPT_mostlikelyobject(image_sequence, text_desc, allow_none)
                output_line = result.choices[0].message.content.split('\n')[-1]
                selected_image_id = int(extract_tag_content(output_line, "output"))
                if selected_image_id < min_valid_id or selected_image_id > len(image_sequence):
                    raise RuntimeError(f'GPT Result Parsing Error.')
                return selected_image_id - 1
            except Exception as e:
                print(f"Error in GPT call. Retrying. Error: {e}")
        raise RuntimeError("GPT call failed.")


def test_gpt_naming():
    gpt_client = GPTClient()

    data = np.load('./test_gpt.npz')
    rgb_images = [data['im0'], data['im1']]
    object_names = gpt_client.obtain_GPT_naming_auto(rgb_images)
    print(f'Object names: {object_names}')
    plot_images(rgb_images)


def test_gpt_object_selection():
    gpt_client = GPTClient()

    data = np.load('./test_gpt.npz')
    rgb_images = [data['im0'], data['im1']]
    plot_images(rgb_images)
    for desc in ['spam can', 'green', 'elephant']:
        mostlikely_id = gpt_client.obtain_GPT_mostlikelyobject_auto(rgb_images, desc)
        print(f'Image most similar to "{desc}"\n\t{mostlikely_id}')


if __name__ == '__main__':
    test_gpt_naming()
    test_gpt_object_selection()
