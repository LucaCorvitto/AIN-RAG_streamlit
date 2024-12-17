from pinecone_embedding import populate_database
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run tests.')
    parser.add_argument('-pdf', '--pdfs-paths-list', action="append", help="Insert the PDFs' paths, follow the same order as the file names")
    args = parser.parse_args()
    pdfs_paths_list = args.pdfs_paths_list

    populate_database(pdfs_paths_list)

if __name__ == "__main__":
    main()