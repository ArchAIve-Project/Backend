import sys
from ASVisual import visualiser

visualiser.main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)