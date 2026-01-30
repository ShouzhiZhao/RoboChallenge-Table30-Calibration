import argparse
import read_data

def main():
    parser = argparse.ArgumentParser(description='Process robot data and fit/view projection models.')
    parser.add_argument('--method', type=str, default='project2d', choices=['project2d', 'pnp'],
                        help='Method to use: project2d (2D rational function) or pnp (3D PnP)')
    
    args = parser.parse_args()
    print(f'Method: {args.method}')
    
    # read_data.main()
    if args.method == 'project2d':
        from project2d import fit, view, export
        fit.main()
        view.main()
        export.main()
    elif args.method == 'pnp':
        from pnp import fit, view, export
        fit.main()
        view.main()
        export.main()

if __name__ == '__main__':
    main()
