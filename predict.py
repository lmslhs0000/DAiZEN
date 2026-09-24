import argparse
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def main(argv=None):
    from service.console import configure_console
    configure_console()
    parser = argparse.ArgumentParser(description='전체 CSV의 차량별 24개월 판매량 예측')
    parser.add_argument('--csv', type=Path, default=ROOT / 'raw/data.csv')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT / 'outputs' / f'forecast_{datetime.now():%Y%m%d_%H%M%S_%f}' / 'forecast.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    print('24개월 판매량을 예측하고 있습니다.', flush=True)
    from service.inference import main as run_inference
    run_inference(['--csv', str(args.csv), '--output', str(output)])
    print(f'결과 폴더: {output.parent.resolve()}')
    print(f'JSON: {output.name} / CSV: forecasft_data.csv / 제외 사유: excluded_vehicles.csv')


if __name__ == '__main__':
    main()
