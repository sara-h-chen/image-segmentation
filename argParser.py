import argparse

parser = argparse.ArgumentParser(description="Processes the image supplied to detect worms. "
                                             "WARNING: Naming convention of file based on image channel must remain consistent")
parser.add_argument("file", help=("The file for processing"))
parser.add_argument("-v", "--verbose", help=("Displays all steps in the process; increases output verbosity"), action="store_true")

# RETURNS ARGUMENT AS args.file
args = parser.parse_args()
# if args.verbose:
#     print args.verbose