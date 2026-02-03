# 🦁 Lynx RF Quant

**Lynx RF Quant** is a lightweight, dual-mode quantitative trading bot powered by **Random Forest**. It supports local backtesting/simulation (**Paper Mode**) and Binance Futures execution (**Live Mode**) through a clean **Streamlit** dashboard.

**Lynx RF Quant** 是一个基于 **随机森林（Random Forest）** 的轻量级量化交易机器人，支持双模式：  
- **Paper Mode（模拟/回测）**：使用本地历史数据进行训练与模拟交易  
- **Live Mode（实盘）**：通过 **Binance Futures API** 执行真实下单  
所有操作均通过简洁的 **Streamlit** 界面完成。

---

## ✨ Features / 功能亮点

- **Dual Mode**: Paper (simulation) & Live (real execution)  
  **双模式**：模拟训练 + 实盘交易
- **Streamlit UI**: One-click start/stop with real-time logs  
  **可视化界面**：一键启动/停止 + 实时日志
- **Random Forest Decision Engine**: Configurable depth, thresholds, horizon  
  **随机森林引擎**：可调深度、阈值、预测窗口
- **Safety Gates**: Min AUC guardrail to prevent low-quality models trading  
  **安全闸门**：AUC 未达标拒绝开单

---

## 🛠️ Installation & Start / 安装与启动

### Prerequisites / 环境要求
- **Python 3.10+**
- **Binance Account** *(Live Mode only / 仅实盘需要)*

### Quick Start / 快速启动

#### 🪟 Windows (Recommended) / Windows 用户（推荐）
Double-click **`run.bat`** in the project folder.  
It will:
- install dependencies from `requirements.txt`
- launch Streamlit dashboard in your browser

双击项目文件夹中的 **`run.bat`**：自动安装依赖并打开网页控制台。

#### 🍎 macOS / 🐧 Linux
Open terminal, go to the project folder, and run:

打开终端，进入项目目录，运行：

```bash
pip install -r requirements.txt
streamlit run app.py
  



---------------------------------------------------------------------------------------------------------------------------
📖 User Guide / 使用流程
1) Select Mode / 选择模式

Paper Trading: simulation (safe, no real money)
选择 模拟训练：安全测试，不涉及真实资金

Live Trading: real execution (requires API Key)
选择 实盘交易：真实下单，需要 API Key

2) Load Data / 投喂数据

Crucial Step / 关键步骤：AI needs historical data to learn.

Drag & drop your BTC .csv files (e.g., from sample_data/) into Load CSV, then click:
📥 1. Process Data

将 BTC 历史 .csv（例如 sample_data/ 里的文件）拖入上传框，然后点击：
📥 1. Process Data

3) Configure (Optional) / 调整参数（可选）

You can adjust leverage, thresholds, or model parameters in the sidebar.


侧边栏可按需调整杠杆、阈值、模型参数。

4) Launch / 启动

(Live Mode only / 仅实盘) Enter Binance API Key & Secret

Click 🚀 2. Start







⚠️ Disclaimer / 免责声明

High Risk Warning / 高风险预警

Educational Use / 仅供教学: This software is for educational and research purposes only and is NOT financial advice.
本软件仅供学习与研究，不构成任何投资建议。

No Guarantee / 无收益保证: Past performance (backtest/simulation) does not guarantee future results.
回测/模拟结果不代表未来收益。

Software Risk / 软件风险: Quant trading involves risks including bugs, API failures, and network latency. The authors are not liable for any financial losses.
量化交易涉及 Bug、API 故障、网络延迟等风险，作者不对资金损失负责。

Use at Your Own Risk / 风险自负: By using this software, you take full responsibility for your trading decisions.
使用本软件即代表你对交易决策承担全部责任。