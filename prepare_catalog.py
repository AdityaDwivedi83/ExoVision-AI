from src.catalog.catalog_parser import CatalogParser

parser = CatalogParser(
    "data/raw/koi_catalog.csv"
)

(
    parser
    .load()
    .filter()
    .add_labels()
)

parser.save(
    "data/raw/targets.csv"
)