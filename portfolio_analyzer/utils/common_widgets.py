"""common_widgets.py"""

import logging
import webbrowser
from typing import Optional

import matplotlib.figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import PathCollection
from matplotlib.patches import Polygon, Rectangle
from PySide6.QtCore import QThread, Signal
from config.app_config import get_config
from data.scraper import get_latest_jgb_1year_rate


logger = logging.getLogger(__name__)


class BrowserLaunchThread(QThread):
    """ブラウザでファイルを開くスレッド"""

    launch_completed = Signal(bool, str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self):
        """スレッド実行"""
        try:
            webbrowser.open(f"file:///{self.file_path}")
            self.launch_completed.emit(True, "ブラウザで表示しました")
        except Exception as e:
            logger.error(f"ブラウザ起動エラー: {e}")
            self.launch_completed.emit(False, f"ブラウザ起動エラー: {str(e)}")


class InterestRateThread(QThread):
    """利子率取得スレッド"""

    rate_fetched = Signal(float)
    fetch_error = Signal(str)

    def run(self):
        """スレッド実行"""
        try:
            result = get_latest_jgb_1year_rate()
            if result:
                _, _, rate, _ = result
                self.rate_fetched.emit(float(rate))
            else:
                self.fetch_error.emit("利回りデータを取得できませんでした")
        except Exception as e:
            logger.error(f"利子率取得エラー: {e}")
            self.fetch_error.emit(str(e))


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for Qt integration
    QtアプリケーションでMatplotlib図を表示するためのキャンバスウィジェット．
    """

    def __init__(self, parent=None, width: float = 10, height: float = 8, dpi: int = 100):
        """初期化"""
        self.figure = matplotlib.figure.Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)

        self.logger = logging.getLogger(__name__)

        # 設定から背景色を取得
        config = get_config()
        colors = config.get_ui_colors()
        bg_color = colors.get('background', '#2b2b2b')
        self.figure.patch.set_facecolor(bg_color)

    def clear_plot(self):
        """プロットを完全にクリア"""
        self.figure.clear()
        self.draw()

    def display_figure(self, fig):
        """外部のfigureを表示"""
        self.figure.clear()

        # 元のfigureの全axesを取得
        src_axes = fig.get_axes()

        if not src_axes:
            self.logger.warning("表示するaxesがありません")
            return

        # axesの合計widthを計算
        total_width = sum(ax.get_position().width for ax in src_axes)
        scale_factor = 1.0

        # 合計widthが0.95未満の場合，スケーリングして画面いっぱいに
        if total_width < 0.95:
            scale_factor = 0.95 / total_width
            self.logger.info(
                f"Scaling axes by {scale_factor:.2f} to fill space "
                f"(total width was {total_width:.2f})"
            )

        # 全てのaxesを位置情報を保持してコピー
        for src_ax in src_axes:
            self._copy_axis(src_ax, scale_factor)

        # 全体タイトル
        if hasattr(fig, '_suptitle') and fig._suptitle:
            self.figure.suptitle(
                fig._suptitle.get_text(),
                color='white',
                fontsize=14,
                fontweight='bold',
                y=0.995
            )

        self.draw()

    def _copy_axis(self, src_ax, scale_factor: float = 1.0):
        """axesをコピーする内部メソッド"""
        # axesの位置情報を取得
        bbox = src_ax.get_position()

        # スケーリング適用
        if scale_factor != 1.0:
            new_width = bbox.width * scale_factor
            new_x0 = bbox.x0 * scale_factor
        else:
            new_width = bbox.width
            new_x0 = bbox.x0

        # projectionを確認
        projection = None
        if hasattr(src_ax, 'name') and src_ax.name == 'polar':
            projection = 'polar'

        # 新しいaxesをスケーリングした位置に作成
        dest_ax = self.figure.add_axes(
            [new_x0, bbox.y0, new_width, bbox.height],
            projection=projection
        )

        # 背景色設定
        dest_ax.set_facecolor(src_ax.get_facecolor())

        # カラーバーaxesかどうかをチェック
        is_colorbar = (bbox.width < 0.1)

        if is_colorbar:
            self._copy_colorbar(src_ax, dest_ax)
            return

        # メインaxesのコピー
        self._copy_main_axes_content(src_ax, dest_ax, projection)

    def _copy_colorbar(self, src_ax, dest_ax):
        """カラーバーaxesをコピー"""
        dest_ax.set_ylabel(src_ax.get_ylabel(), color='white', fontsize=10)
        dest_ax.tick_params(colors='white')
        dest_ax.set_ylim(src_ax.get_ylim())

        try:
            dest_ax.set_yticks(src_ax.get_yticks())
            labels = [t.get_text() for t in src_ax.get_yticklabels()]
            if labels:
                dest_ax.set_yticklabels(labels, color='white', fontsize=9)
        except (AttributeError, ValueError, TypeError) as e:
            self.logger.debug(f"Failed to copy y-axis labels: {e}")

        # カラーバーのimageをコピー
        for image in src_ax.get_images():
            try:
                array = image.get_array()
                extent = image.get_extent()
                cmap = image.get_cmap()
                alpha = image.get_alpha() if image.get_alpha() is not None else 1.0
                dest_ax.imshow(array, cmap=cmap, aspect='auto', extent=extent, alpha=alpha)
            except (AttributeError, ValueError, TypeError) as e:
                self.logger.debug(f"Failed to copy image: {e}")

    def _copy_main_axes_content(self, src_ax, dest_ax, projection: Optional[str]):
        """メインaxesのコンテンツをコピー"""
        # imagesをコピー
        for image in src_ax.get_images():
            try:
                array = image.get_array()
                extent = image.get_extent()
                cmap = image.get_cmap()
                alpha = image.get_alpha() if image.get_alpha() is not None else 1.0
                dest_ax.imshow(array, cmap=cmap, aspect='auto', extent=extent, alpha=alpha)
            except Exception as e:
                self.logger.warning(f"Image copy error: {e}")

        # 線をコピー
        for line in src_ax.get_lines():
            dest_ax.plot(
                line.get_xdata(), line.get_ydata(),
                color=line.get_color(),
                linewidth=line.get_linewidth(),
                linestyle=line.get_linestyle(),
                marker=line.get_marker(),
                label=line.get_label()
            )

        # collectionsをコピー
        self._copy_collections(src_ax, dest_ax)

        # テキストをコピー
        for text in src_ax.texts:
            try:
                dest_ax.text(
                    text.get_position()[0], text.get_position()[1], text.get_text(),
                    color=text.get_color(), fontsize=text.get_fontsize(),
                    ha=text.get_ha(), va=text.get_va(), weight=text.get_weight()
                )
            except (AttributeError, ValueError, TypeError) as e:
                self.logger.debug(f"Failed to copy text: {e}")

        # パッチをコピー
        self._copy_patches(src_ax, dest_ax)

        # タイトルとラベル
        self._copy_labels_and_title(src_ax, dest_ax, projection)

        # グリッドと凡例
        self._copy_grid_and_legend(src_ax, dest_ax, projection)

    def _copy_collections(self, src_ax, dest_ax):
        """コレクションをコピー"""
        for collection in src_ax.collections:
            try:
                if isinstance(collection, PathCollection):
                    offsets = collection.get_offsets()
                    sizes = collection.get_sizes()
                    facecolors = collection.get_facecolors()
                    edgecolors = collection.get_edgecolors()
                    linewidths = collection.get_linewidths()
                    alpha = collection.get_alpha()
                    dest_ax.scatter(
                        offsets[:, 0], offsets[:, 1], s=sizes, c=facecolors,
                        edgecolors=edgecolors, linewidths=linewidths, alpha=alpha
                    )
                else:
                    paths = collection.get_paths() if hasattr(collection, 'get_paths') else None
                    if paths:
                        for path in paths:
                            vertices = path.vertices
                            if len(vertices) > 0:
                                facecolor = (collection.get_facecolor()[0]
                                           if len(collection.get_facecolor()) > 0 else 'blue')
                                edgecolor = (collection.get_edgecolor()[0]
                                           if len(collection.get_edgecolor()) > 0 else 'none')
                                dest_ax.fill(
                                    vertices[:, 0], vertices[:, 1],
                                    facecolor=facecolor,
                                    alpha=collection.get_alpha() or 0.15,
                                    edgecolor=edgecolor
                                )
            except Exception as e:
                self.logger.warning(f"Collection copy error: {e}")

    def _copy_patches(self, src_ax, dest_ax):
        """パッチをコピー"""
        for patch in src_ax.patches:
            try:
                if isinstance(patch, Rectangle):
                    dest_ax.add_patch(Rectangle(
                        patch.get_xy(), patch.get_width(), patch.get_height(),
                        facecolor=patch.get_facecolor(), edgecolor=patch.get_edgecolor(),
                        linewidth=patch.get_linewidth(), alpha=patch.get_alpha()
                    ))
                elif isinstance(patch, Polygon):
                    dest_ax.add_patch(Polygon(
                        patch.get_xy(), facecolor=patch.get_facecolor(),
                        edgecolor=patch.get_edgecolor(), linewidth=patch.get_linewidth(),
                        alpha=patch.get_alpha()
                    ))
            except (AttributeError, ValueError, TypeError) as e:
                self.logger.debug(f"Failed to copy patch: {e}")

    def _copy_labels_and_title(self, src_ax, dest_ax, projection: Optional[str]):
        """ラベルとタイトルをコピー"""
        dest_ax.set_title(src_ax.get_title(), color='white', fontsize=12, fontweight='bold')

        if projection != 'polar':
            dest_ax.set_xlabel(src_ax.get_xlabel(), color='white', fontsize=10)
            dest_ax.set_ylabel(src_ax.get_ylabel(), color='white', fontsize=10)
            try:
                dest_ax.set_xlim(src_ax.get_xlim())
                dest_ax.set_ylim(src_ax.get_ylim())
            except (AttributeError, ValueError, TypeError) as e:
                self.logger.debug(f"Failed to set axis limits: {e}")

        dest_ax.tick_params(colors='white')

    def _copy_grid_and_legend(self, src_ax, dest_ax, projection: Optional[str]):
        # 設定から色を取得
        config = get_config()
        colors = config.get_ui_colors()
        bg_color = colors.get('background', '#2b2b2b')
        grid_color = colors.get('grid', '#444444')
        
        """グリッドと凡例をコピー"""
        if projection == 'polar':
            try:
                dest_ax.set_xticks(src_ax.get_xticks())
                dest_ax.set_xticklabels(
                    [t.get_text() for t in src_ax.get_xticklabels()],
                    color='white', fontsize=11
                )
                dest_ax.set_ylim(src_ax.get_ylim())
                dest_ax.set_yticks(src_ax.get_yticks())
                dest_ax.set_yticklabels(
                    [t.get_text() for t in src_ax.get_yticklabels()],
                    color='white', fontsize=9, alpha=0.7
                )
            except (AttributeError, ValueError, TypeError) as e:
                self.logger.debug(f"Failed to set polar ticks: {e}")
            dest_ax.grid(True, color=grid_color, linestyle='--', linewidth=0.5, alpha=0.5)
        else:
            dest_ax.grid(True, alpha=0.2, color=grid_color)
            if src_ax.get_xticklabels():
                try:
                    dest_ax.set_xticks(src_ax.get_xticks())
                    labels = [t.get_text() for t in src_ax.get_xticklabels()]
                    if labels:
                        dest_ax.set_xticklabels(labels, rotation=0, color='white', fontsize=9)
                except (AttributeError, ValueError, TypeError) as e:
                    self.logger.debug(f"Failed to set x-axis labels: {e}")
        
        # 凡例
        legend = src_ax.get_legend()
        if legend:
            handles, labels = src_ax.get_legend_handles_labels()
            if handles:
                if projection == 'polar':
                    dest_ax.legend(
                        handles, labels, loc='upper right', bbox_to_anchor=(1.3, 1.1),
                        facecolor=bg_color, edgecolor='white',
                        labelcolor='white', fontsize=10
                    )
                else:
                    dest_ax.legend(
                        handles, labels, facecolor=bg_color, edgecolor='white',
                        labelcolor='white', fontsize=9
                    )


__all__ = ['BrowserLaunchThread', 'InterestRateThread', 'MatplotlibCanvas']