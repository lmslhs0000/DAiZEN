import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def predict(csv_path=None, output=None):
    from service.console import configure_console
    configure_console()
    csv_path = Path(csv_path) if csv_path is not None else ROOT / 'raw/data.csv'
    output = Path(output) if output is not None else ROOT / 'outputs' / 'forecast' / 'forecast.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    print('24개월 판매량을 예측하고 있습니다.', flush=True)
    from service.inference import main as run_inference
    result = run_inference(['--csv', str(csv_path), '--output', str(output)])
    print(f'결과 폴더: {output.parent.resolve()}')
    print(f'JSON: {output.name} / CSV: forecasft_data.csv')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='전체 CSV의 차량별 24개월 판매량 예측')
    parser.add_argument('--csv', type=Path, default=ROOT / 'raw/data.csv')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    return predict(args.csv, args.output)


if __name__ == '__main__':
    result = main()
