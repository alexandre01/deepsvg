from concurrent import futures
import os
from argparse import ArgumentParser
import logging
from tqdm import tqdm
import glob

from deepsvg.svglib.svg import SVG


def preprocess_svg(svg_file, output_folder):
    filename = os.path.splitext(os.path.basename(svg_file))[0]

    svg = SVG.load_svg(svg_file)
    svg.fill_(False)
    svg.normalize()
    svg.zoom(0.9)
    svg.canonicalize()
    svg = svg.simplify_heuristic()

    svg.save_svg(os.path.join(output_folder, f"{filename}.svg"))


def main(args):
    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        svg_files = glob.glob(os.path.join(args.data_folder, "*.svg"))

        with tqdm(total=len(svg_files)) as pbar:
            preprocess_requests = [executor.submit(preprocess_svg, svg_file, args.output_folder)
                                    for svg_file in svg_files]

            for _ in futures.as_completed(preprocess_requests):
                pbar.update(1)

    logging.info("SVG Preprocessing complete.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--data_folder", default=os.path.join("dataset", "svgs"))
    parser.add_argument("--output_folder", default=os.path.join("dataset", "svgs_simplified"))
    parser.add_argument("--workers", default=4, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.output_folder): os.makedirs(args.output_folder)

    main(args)
