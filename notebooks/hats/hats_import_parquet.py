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


def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument("input", help="Input catalog", type=Path)
    parser.add_argument("output", help="Root of the output directory", type=Path)
    parser.add_argument("-n", "--name", required=True, help="catalog name")
    return parser.parse_args(argv)


def hats_import_main_catalog(
        input_path: Path,
        output_dir: Path,
        output_name: str,
        ra_column: str,
        dec_column: str,
        dask_kwargs: dict | None = None,
):      
    input_file_list = sorted(input_path.glob("*.parquet"))
    
    args = ImportArguments(
        ra_column=ra_column,
        dec_column=dec_column,
        input_file_list=input_file_list,
        file_reader=ParquetPyarrowReader(),
        output_artifact_name=output_name,
        output_path=output_dir,
        pixel_threshold=100_000,
        use_healpix_29=True,
        add_healpix_29=False,
    )

    with Client(**(dask_kwargs or {})) as client:
        logging.info(str(client))
        pipeline_with_client(args, client)


def hats_import_margin_catalog(
        main_catalog_path: Path,
        margin_threshold_arcsec: int,
        dask_kwargs: dict | None = None,
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

    with Client(**(dask_kwargs or {})) as client:
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


def hats_import_parquet(
        inp: Path,
        out: Path,
        name: str,
        ra_column: str = '_RAJ2000',
        dec_column: str = '_DEJ2000',
        dask_kwargs: dict | None = None,
):
    hats_import_main_catalog(
        inp,
        out,
        name,
        ra_column=ra_column,
        dec_column=dec_column,
        dask_kwargs=dask_kwargs,
    )
    hats_import_margin_catalog(out / name, 10, dask_kwargs=dask_kwargs)


def main(argv=None):
    args = parse_args(argv)
    inp = args.input
    out = args.output
    name = args.name

    hats_reimport(inp, out, name)


if __name__ == '__main__':
    main()
