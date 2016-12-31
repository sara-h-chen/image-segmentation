import argparse

parser = argparse.ArgumentParser(description="Carries out image segmentation based on shape and blob detection")
parser.add_argument('--foo', help=('foo help'))

# >>> parser.add_argument('integers', metavar='N', type=int, nargs='+',
# ...                     help='an integer for the accumulator')
# >>> parser.add_argument('--sum', dest='accumulate', action='store_const',
# ...                     const=sum, default=max,
# ...                     help='sum the integers (default: find the max)')
# Later, calling parse_args() will return an object with two attributes, integers and accumulate. The integers attribute will be a list of one or more ints, and the accumulate attribute will be either the sum() function, if --sum was specified at the command line, or the max() function if it was not.