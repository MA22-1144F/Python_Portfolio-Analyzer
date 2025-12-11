"""
アイコン生成スクリプト
app_icon.pngから複数サイズを含む.icoファイルを生成します
"""

from PIL import Image
from pathlib import Path

def generate_ico_from_png(png_path, ico_path):
    """PNGファイルから複数サイズを含む.icoファイルを生成"""

    # PNGファイルを開く
    img = Image.open(png_path)

    # RGBA形式に変換
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # 生成するサイズリスト（Windowsで使用される標準サイズ）
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

    # 各サイズの画像を生成
    icon_images = []
    for size in sizes:
        # リサイズ（高品質）
        resized = img.resize(size, Image.Resampling.LANCZOS)
        icon_images.append(resized)

    # .icoファイルとして保存
    img.save(ico_path, format='ICO', sizes=sizes)

    print(f"[OK] Generated: {ico_path}")
    print(f"  Sizes: {', '.join([f'{s[0]}x{s[1]}' for s in sizes])}")

def main():
    # パスを設定
    script_dir = Path(__file__).parent
    assets_dir = script_dir / 'assets' / 'icons'

    png_path = assets_dir / 'app_icon.png'
    ico_path = assets_dir / 'app_icon.ico'

    if not png_path.exists():
        print(f"Error: {png_path} not found")
        return 1

    print(f"Generating multi-size .ico from {png_path}...")

    try:
        generate_ico_from_png(png_path, ico_path)
        print("\nSuccess! Icon file generated successfully.")
        print(f"\nBackup the old icon if needed, then rebuild the exe:")
        print("  build_exe.bat")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
