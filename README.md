
# Portfolio-Analyzer

<img src="portfolio_analyzer/assets/icons/app_icon.png" alt="icon" width="200">
クロスアセット対応ポートフォリオ最適化・分析ソフトウェア

## 概要

Portfolio-Analyzer(P-A)は，現代ポートフォリオ理論(MPT)およびポスト・モダン・ポートフォリオ理論(PMPT)に基づくポートフォリオ分析ツールです．

## 主な機能

- **効率的フロンティア分析**：平均分散最適化(MVO)
- **下方偏差フロンティア分析**：ポスト・モダン・ポートフォリオ理論(PMPT)
- **相関行列分析**：資産間の相関構造を可視化
- **証券市場線分析**：CAPMに基づくβ値推定
- **リスク指標**：VaR、CVaR、最大ドローダウン等

## ダウンロード

[Releases](../../releases) から最新版をダウンロードしてください．

| OS | ファイル |
|----|----------|
| Windows | `PortfolioAnalyzer.exe` |
| macOS | `PortfolioAnalyzer-macOS.dmg` |

## ソースから実行
```bash
git clone https://github.com/MA22-1144F/Portfolio-Analyzer.git
cd Portfolio-Analyzer/portfolio_analyzer
pip install -r requirements.txt
python main.py
```

## 動作環境

- Windows 10/11
- macOS 10.13以降
- Python 3.10以上(ソースから実行する場合)

## ライセンス

MIT License

## 作者

MA22-1144F: ma221144(at)senshu-u.jp

## 関連

本ソフトウェアは，専修大学経営学部 2025年度卒業論文の成果物です．
