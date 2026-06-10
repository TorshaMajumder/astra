#!/usr/bin/env python3

import logging
from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree

from dask.distributed import Client
from hats import read_hats
from hats.catalog import Catalog
from hats_import.catalog.arguments import ImportArguments
from hats_import.catalog.file_readers import ParquetPyarrowReader
from hats_import.margin_cache.margin_cache_arguments import MarginCacheArguments
from hats_import.pipeline import pipeline_with_client
# from nested_pandas import NestedDtype as _


def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument("input", help="Input HATS catalog dir", type=Path)
    parser.add_argument("output", help="Root of the output directory", type=Path)
    parser.add_argument("-n", "--new-name", default=None, help="rename catalog")
    return parser.parse_args(argv)


def get_hats_catalog(inp: Path):
    properties = next(inp.rglob("properties"))
    catalog_path = properties.parent
    return read_hats(catalog_path)


def hats_import_main_catalog(
        input_catalog_path: Path,
        output_dir: Path,
        output_name: str | None,
        catalog: Catalog,
):
    column_names = catalog.schema.names
    column_names.remove('Dir')
    column_names.remove('Norder')
    column_names.remove('Npix')
    
    name = catalog.catalog_name if output_name is None else output_name
    
    input_file_list = sorted(input_catalog_path.rglob("*.parquet"))
    
    args = ImportArguments(
        ra_column=catalog.catalog_info.ra_column,
        dec_column=catalog.catalog_info.dec_column,
        input_file_list=input_file_list,
        file_reader=ParquetPyarrowReader(column_names=column_names),
        output_artifact_name=name,
        output_path=output_dir,
        pixel_threshold=100_000,
        use_healpix_29=True,
        add_healpix_29=False,
    )

    with Client() as client:
        logging.info(str(client))
        pipeline_with_client(args, client)

    return name


def hats_import_margin_catalog(
    main_catalog_path: Path,
    margin_threshold_arcsec: int,
) -> bool:
    catalog = read_hats(main_catalog_path)

    margin_catalog_name = f"{catalog.catalog_name}_{margin_threshold_arcsec}arcsec"
    args = MarginCacheArguments(
        input_catalog_path=main_catalog_path,
        output_path=main_catalog_path.parent,
        margin_threshold=float(margin_threshold_arcsec),
        output_artifact_name=margin_catalog_name,
        fine_filtering=False,
    )

    with Client() as client:
        logging.info(str(client))
        try:
            pipeline_with_client(args, client)
        except ValueError as e:
            if "Margin cache contains no rows" in str(e):
                margin_path = main_catalog_path.parent / margin_catalog_name
                rmtree(margin_path)
                return False
            raise
    return True


def hats_reimport(inp: Path, out: Path, name: str | None):
    hats_catalog = get_hats_catalog(inp)
    name = hats_import_main_catalog(inp, out, name, hats_catalog)
    hats_import_margin_catalog(out / name, 10)


def main(argv=None):
    args = parse_args(argv)
    inp = args.input
    out = args.output
    name = args.new_name

    hats_reimport(inp, out, name)


if __name__ == '__main__':
    main()
