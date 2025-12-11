"""
Mac用アイコン生成スクリプト
app_icon.pngから.icnsファイルを生成します
"""

from PIL import Image
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil


def generate_icns_from_png(png_path, icns_path):
    """PNGファイルから.icnsファイルを生成（macOS専用）"""

    # macOSでのみ実行可能
    if sys.platform != 'darwin':
        print("Warning: .icns generation works best on macOS")
        print("On non-macOS systems, we'll create an iconset structure that can be converted on macOS")

    # PNGファイルを開く
    img = Image.open(png_path)

    # RGBA形式に変換
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # 一時的なiconsetディレクトリを作成
    iconset_name = 'AppIcon.iconset'
    iconset_dir = png_path.parent / iconset_name

    # 既存のiconsetディレクトリがあれば削除
    if iconset_dir.exists():
        shutil.rmtree(iconset_dir)

    iconset_dir.mkdir()

    # macOSアイコンの標準サイズと対応するファイル名
    icon_sizes = [
        (16, 'icon_16x16.png'),
        (32, 'icon_16x16@2x.png'),
        (32, 'icon_32x32.png'),
        (64, 'icon_32x32@2x.png'),
        (128, 'icon_128x128.png'),
        (256, 'icon_128x128@2x.png'),
        (256, 'icon_256x256.png'),
        (512, 'icon_256x256@2x.png'),
        (512, 'icon_512x512.png'),
        (1024, 'icon_512x512@2x.png'),
    ]

    print(f"Generating iconset from {png_path}...")

    # 各サイズの画像を生成
    for size, filename in icon_sizes:
        resized = img.resize((size, size), Image.Resampling.LANCZOS)
        output_path = iconset_dir / filename
        resized.save(output_path, 'PNG')
        print(f"  [OK] Created {filename} ({size}x{size})")

    # macOSの場合、iconutilを使って.icnsに変換
    if sys.platform == 'darwin':
        try:
            print(f"\nConverting iconset to .icns...")
            subprocess.run(
                ['iconutil', '-c', 'icns', str(iconset_dir), '-o', str(icns_path)],
                check=True
            )
            print(f"[OK] Generated: {icns_path}")

            # iconsetディレクトリを削除
            shutil.rmtree(iconset_dir)
            print(f"[OK] Cleaned up temporary iconset directory")

        except subprocess.CalledProcessError as e:
            print(f"Error converting to .icns: {e}")
            print(f"Iconset directory preserved at: {iconset_dir}")
            return False
        except FileNotFoundError:
            print("Error: iconutil command not found")
            print("This command is only available on macOS")
            return False
    else:
        print(f"\nIconset created at: {iconset_dir}")
        print("To convert to .icns on macOS, run:")
        print(f"  iconutil -c icns {iconset_dir} -o {icns_path}")
        print("\nFor now, copying PNG as fallback icon...")
        # 非macOSの場合、512x512のPNGをそのままコピー（PyInstallerはPNGも受け付ける）
        fallback = img.resize((512, 512), Image.Resampling.LANCZOS)
        fallback.save(icns_path.with_suffix('.png'), 'PNG')
        print(f"[OK] Created fallback PNG: {icns_path.with_suffix('.png')}")

    return True


def main():
    # パスを設定
    script_dir = Path(__file__).parent
    assets_dir = script_dir / 'assets' / 'icons'

    png_path = assets_dir / 'app_icon.png'
    icns_path = assets_dir / 'app_icon.icns'

    if not png_path.exists():
        print(f"Error: {png_path} not found")
        return 1

    print("=" * 60)
    print("Portfolio Analyzer - macOS Icon Generator")
    print("=" * 60)
    print()

    try:
        if generate_icns_from_png(png_path, icns_path):
            print("\n" + "=" * 60)
            print("Success! macOS icon generated successfully.")
            print("=" * 60)
            if sys.platform == 'darwin':
                print("\nNext steps:")
                print("  1. Run: ./build_mac.sh")
                print("  2. Find the app in: dist/PortfolioAnalyzer.app")
            return 0
        else:
            print("\nPartially completed. See messages above.")
            return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
