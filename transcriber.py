from segmentation import segment_image, TextBubble, TextDetector
import argparse
import torch
import logging
import tqdm
from typing import Literal
import os
from manga_ocr import MangaOcr


IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]


class Transcriber:
    def __list_images(self, image_dir: str) -> list[str]:
        # don't do recursive search so far
        return [
            os.path.join(image_dir, fn)
            for fn in os.listdir(image_dir)
            if os.path.splitext(fn)[1].lower() in IMAGE_SUFFIXES
        ]

    def __init__(
        self,
        model_path: str,
        sort_mode: Literal["right-to-left", "left-to-right"],
        device: str = "cpu",
    ):
        self.model = TextDetector(
            model_path, input_size=1024, device=device, act="leaky"
        )
        self.sort_ltr = sort_mode == "left-to-right"
        self.mocr = MangaOcr()

    def __transcribe(self, image_path: str) -> list[str]:
        bubbles = segment_image(self.model, image_path)
        logging.debug(f"Transcribed {len(bubbles)} bubbles in {image_path}")

        # sort bubbles first by y, then by
        #   x + width if right-to-left(Japanese manga is sorted this way)
        #   x if left-to-right(English comics are sorted this way)
        def comp(bubble: TextBubble):
            return bubble.xyxy[1], (
                bubble.xyxy[0] if self.sort_ltr else -bubble.xyxy[2]
            )

        bubbles.sort(key=comp)
        for bubble in bubbles:
            bubble.text = self.mocr(bubble.image)
        return [bubble.text for bubble in bubbles]

    def transcribe(self, image_dir: str, output: str):
        if os.path.isdir(output):
            output = os.path.join(output, "transcription.txt")
        images = self.__list_images(image_dir)

        logging.info(f"Transcribing {len(images)} images to {output}")
        output_temp = output + ".tmp"
        done = set()
        if not os.path.exists(output_temp):
            # create the file
            open(output_temp, "w").close()
        else:
            with open(output_temp, "r") as f:
                done = set([line.strip() for line in f.readlines()])
        with open(output, "a") as f, open(output_temp, "a") as f_temp:
            for image_path in tqdm.tqdm(images):
                if os.path.basename(image_path) in done:
                    continue
                image_fn = os.path.basename(image_path)
                f.write(f"{image_fn}\n")
                for text in self.__transcribe(image_path):
                    text = text.strip().replace("\n", " ")
                    if len(text) == 0:
                        continue
                    text = text
                    f.write(text + "\n")
                done.add(image_fn)
                f_temp.write(image_fn + "\n")
        logging.info(f"Transcription complete, saved to {output}")
        os.remove(output_temp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="data/comictextdetector.pt")
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Directory of images to process",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Directory or file to save the output text to",
        default="transcriber.txt",
    )
    parser.add_argument(
        "--sort_mode",
        choices=["right-to-left", "left-to-right"],
        default="right-to-left",
        help="Sort mode for text bubbles, default: right-to-left",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level, default: INFO",
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    transcriber = Transcriber(args.model_path, args.sort_mode, device)
    transcriber.transcribe(args.image_dir, args.output)
