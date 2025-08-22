# Excel多表合并工具 - AI智能匹配表头

## 项目简介

这是一个基于Python + DeepSeek LLM的Excel多表合并工具，能够智能识别语义相同但表述不同的字段名，实现异构表格的秒级对齐合并。

### 核心功能

- 🤖 **AI智能匹配**: 使用DeepSeek LLM识别相似字段（如：订单ID、ID_订单、订单编号、销售单号）
- 🔄 **模糊匹配备选**: 当AI分析失败时，自动回退到模糊匹配算法
- 📊 **批量处理**: 支持同时合并多个Excel文件
- 🧹 **数据清理**: 自动去重、处理数据冲突
- 📈 **详细报告**: 生成完整的合并报告和字段映射信息
- ⚙️ **配置灵活**: 支持配置文件自定义各种参数

### 应用场景

- 多分公司销售报表合并
- 不同系统导出数据整合
- 历史数据迁移和标准化
- 数据仓库ETL预处理

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements_excel_merger.txt
```

### 2. 配置API密钥

编辑 `config_excel_merger.json` 文件，设置您的DeepSeek API密钥：

```json
{
  "deepseek_api": {
    "api_key": "sk-your-actual-deepseek-api-key",
    "base_url": "https://api.deepseek.com"
  }
}
```

### 3. 运行演示

```bash
# 创建示例文件并运行演示
python excel_merger_enhanced.py --demo

# 或者只创建示例文件
python excel_merger_enhanced.py --create-sample
```

### 4. 合并自己的文件

```bash
# 合并指定的Excel文件
python excel_merger_enhanced.py --files file1.xlsx file2.xlsx file3.xlsx

# 指定输出目录
python excel_merger_enhanced.py --files *.xlsx --output ./results

# 使用自定义配置文件
python excel_merger_enhanced.py --files *.xlsx --config my_config.json
```

## 详细使用说明

### 程序文件说明

- `excel_merger_ai.py`: 基础版合并工具
- `excel_merger_enhanced.py`: 增强版合并工具（推荐使用）
- `config_excel_merger.json`: 配置文件
- `requirements_excel_merger.txt`: 依赖包列表

### 配置文件详解

```json
{
  "deepseek_api": {
    "api_key": "您的API密钥",
    "base_url": "API基础URL",
    "model": "deepseek-chat",
    "temperature": 0.1,
    "max_tokens": 2000,
    "timeout": 30
  },
  "merge_settings": {
    "remove_duplicates": true,      // 是否移除重复数据
    "add_source_info": true,        // 是否添加数据源信息
    "case_sensitive": false,        // 字段匹配是否区分大小写
    "auto_detect_encoding": true    // 是否自动检测文件编码
  },
  "field_matching": {
    "similarity_threshold": 0.8,    // 模糊匹配相似度阈值
    "use_ai_analysis": true,        // 是否使用AI分析
    "fallback_to_fuzzy_match": true, // 是否回退到模糊匹配
    "common_field_patterns": {      // 预定义字段模式
      "order_id": ["订单ID", "订单编号", "ID_订单", "销售单号"],
      "customer_name": ["客户名称", "客户名", "客户全称", "公司名称"]
    }
  }
}
```

### 命令行参数

```bash
python excel_merger_enhanced.py [选项]

选项:
  --files FILE1 FILE2 ...    要合并的Excel文件路径
  --config CONFIG_FILE       配置文件路径 (默认: config_excel_merger.json)
  --output OUTPUT_DIR        输出目录 (默认: ./output)
  --demo                     运行演示模式
  --create-sample            创建示例文件
  --help                     显示帮助信息
```

### 输出文件说明

程序运行后会在输出目录生成以下文件：

1. **merged_data_YYYYMMDD_HHMMSS.xlsx**: 合并后的Excel文件
2. **merge_report_YYYYMMDD_HHMMSS.json**: 详细的合并报告
3. **field_mapping_YYYYMMDD_HHMMSS.json**: 字段映射关系

### 合并报告示例

```json
{
  "merge_info": {
    "merge_time": "2024-01-29T10:30:00",
    "processing_time_seconds": 15.6,
    "tool_version": "2.0.0"
  },
  "statistics": {
    "input_stats": {
      "files_processed": 4,
      "total_input_rows": 156
    },
    "output_stats": {
      "total_output_rows": 148,
      "duplicates_removed": 8,
      "data_columns": 6
    }
  },
  "field_mapping": {
    "订单ID": ["订单ID", "ID_订单", "销售单号", "order_id"],
    "客户名称": ["客户名称", "客户全称", "公司名称", "customer_name"]
  }
}
```

## 高级功能

### 1. 自定义字段映射规则

在配置文件中添加常用的字段模式，提高匹配准确性：

```json
"common_field_patterns": {
  "product_code": ["产品编码", "商品代码", "SKU", "product_id"],
  "quantity": ["数量", "qty", "amount", "count"],
  "unit_price": ["单价", "价格", "unit_price", "price"]
}
```

### 2. 批量处理模式

```python
# 编程方式使用
from excel_merger_enhanced import ExcelMergerEnhanced

merger = ExcelMergerEnhanced('config.json')
merger.load_excel_files(['file1.xlsx', 'file2.xlsx', 'file3.xlsx'])
merger.analyze_and_merge()
result_files = merger.save_results('./output')
```

### 3. 错误处理和恢复

程序具备完善的错误处理机制：

- API调用失败时自动回退到模糊匹配
- 文件读取错误时跳过问题文件继续处理
- 详细的错误日志记录

## 性能优化

### 1. 缓存机制

- 字段映射结果会被缓存，避免重复分析
- 相同字段组合的映射关系会被复用

### 2. 批处理优化

- 支持大文件分块处理
- 内存使用优化，避免OOM错误

### 3. API调用优化

- 智能重试机制
- 请求频率控制
- 错误统计和监控

## 常见问题

### Q1: API密钥如何获取？

A: 访问 [DeepSeek官网](https://www.deepseek.com) 注册账号并获取API密钥。

### Q2: 支持哪些Excel格式？

A: 支持 .xlsx、.xls 格式，推荐使用 .xlsx 格式。

### Q3: 如何处理大文件？

A: 程序会自动处理大文件，如果遇到内存问题，可以：
- 分批处理文件
- 增加系统内存
- 使用数据采样模式

### Q4: 字段匹配不准确怎么办？

A: 可以通过以下方式改进：
- 在配置文件中添加自定义字段模式
- 调整相似度阈值
- 手动编辑字段映射文件后重新合并

### Q5: 如何处理中英文混合字段？

A: 程序已内置中英文混合处理逻辑，会自动识别：
- 中英文对照（如：客户名称 / Customer Name）
- 简写和全称（如：ID / 标识符）
- 不同命名风格（如：snake_case / camelCase）

## 技术架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Excel Files   │───▶│  Field Analyzer  │───▶│  Data Merger    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                       ┌──────────────┐         ┌──────────────┐
                       │ DeepSeek API │         │ Result Files │
                       └──────────────┘         └──────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ Fuzzy Match  │
                       │  (Fallback)  │
                       └──────────────┘
```

## 更新日志

### v1.0.0 (2025-08-22)
- 基础版本发布
- 支持DeepSeek LLM字段匹配
- 基本的Excel文件合并功能
- 新增模糊匹配备选方案
- 添加配置文件支持
- 增强错误处理和恢复机制
- 优化API调用和缓存机制
- 添加详细的合并报告
- 支持命令行参数

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个工具！

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 使用本工具前请确保您有合法的数据处理权限，并遵守相关的数据保护法规。