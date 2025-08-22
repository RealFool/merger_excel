#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel多表合并工具 - AI智能匹配表头
使用DeepSeek LLM识别语义相同但表述不同的字段，实现异构表格智能对齐合并

功能特点：
1. 支持多个Excel文件批量合并
2. AI智能识别相似字段名（如：订单ID、ID_订单、订单编号、销售单号）
3. 自动对齐表头，处理异构表格
4. 支持数据去重和冲突处理
5. 生成合并报告

作者：AI Assistant
日期：2025-01-29
"""

import os
import json
import pandas as pd
import requests
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime
import hashlib
from collections import defaultdict
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('excel_merger.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeepSeekAPI:
    """DeepSeek API调用类"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com"):
        self.api_key = api_key or "sk-your-deepseek-api-key"  # 请替换为实际的API Key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def call_api(self, messages: List[Dict], model: str = "deepseek-chat") -> str:
        """调用DeepSeek API"""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2000
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"API调用失败: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"API调用异常: {e}")
            return ""

class FieldMatcher:
    """字段匹配器 - 使用AI识别相似字段"""
    
    def __init__(self, deepseek_api: DeepSeekAPI):
        self.api = deepseek_api
        self.field_mapping_cache = {}
    
    def analyze_field_similarity(self, field_groups: List[List[str]]) -> Dict[str, List[str]]:
        """分析字段相似性，返回字段映射关系"""
        
        # 构建所有字段列表
        all_fields = []
        for group in field_groups:
            all_fields.extend(group)
        
        # 去重
        unique_fields = list(set(all_fields))
        
        if len(unique_fields) <= 1:
            return {unique_fields[0]: unique_fields} if unique_fields else {}
        
        # 构建AI提示词
        prompt = self._build_field_analysis_prompt(unique_fields)
        
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的数据分析师，擅长识别表格中语义相同但表述不同的字段名。请仔细分析字段名的语义含义，将相同含义的字段归为一组。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # 调用API
        response = self.api.call_api(messages)
        
        # 解析响应
        field_mapping = self._parse_field_mapping_response(response, unique_fields)
        
        logger.info(f"字段映射结果: {field_mapping}")
        return field_mapping
    
    def _build_field_analysis_prompt(self, fields: List[str]) -> str:
        """构建字段分析提示词"""
        fields_text = "\n".join([f"{i+1}. {field}" for i, field in enumerate(fields)])
        
        prompt = f"""
请分析以下字段名，识别语义相同但表述不同的字段，并将它们归为一组。

字段列表：
{fields_text}

分析要求：
1. 识别表示相同业务含义的字段（如：订单ID、ID_订单、订单编号、销售单号都表示订单标识）
2. 考虑常见的业务字段：客户信息、订单信息、产品信息、金额、时间等
3. 忽略大小写、下划线、空格等格式差异
4. 考虑中英文混用、简写、全称等情况

请以JSON格式返回结果，格式如下：
{{
  "group_1": ["字段1", "字段2", "字段3"],
  "group_2": ["字段4", "字段5"],
  "group_3": ["字段6"]
}}

其中每个group代表一组语义相同的字段，group的命名请使用最通用的字段名。
"""
        return prompt
    
    def _parse_field_mapping_response(self, response: str, original_fields: List[str]) -> Dict[str, List[str]]:
        """解析AI响应，提取字段映射关系"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                mapping = json.loads(json_str)
                
                # 验证映射结果
                mapped_fields = set()
                for group_fields in mapping.values():
                    mapped_fields.update(group_fields)
                
                # 添加未映射的字段
                for field in original_fields:
                    if field not in mapped_fields:
                        mapping[field] = [field]
                
                return mapping
        except Exception as e:
            logger.warning(f"解析AI响应失败: {e}")
        
        # 如果解析失败，返回默认映射（每个字段自成一组）
        return {field: [field] for field in original_fields}

class ExcelMerger:
    """Excel合并器主类"""
    
    def __init__(self, deepseek_api_key: str = None):
        self.api = DeepSeekAPI(deepseek_api_key)
        self.field_matcher = FieldMatcher(self.api)
        self.dataframes = []
        self.file_info = []
        self.merged_df = None
        self.merge_report = {}
    
    def load_excel_files(self, file_paths: List[str]) -> bool:
        """加载多个Excel文件"""
        logger.info(f"开始加载 {len(file_paths)} 个Excel文件")
        
        self.dataframes = []
        self.file_info = []
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"文件不存在: {file_path}")
                    continue
                
                # 读取Excel文件
                df = pd.read_excel(file_path)
                
                if df.empty:
                    logger.warning(f"文件为空: {file_path}")
                    continue
                
                self.dataframes.append(df)
                self.file_info.append({
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'rows': len(df),
                    'columns': list(df.columns),
                    'column_count': len(df.columns)
                })
                
                logger.info(f"成功加载: {os.path.basename(file_path)} ({len(df)}行, {len(df.columns)}列)")
                
            except Exception as e:
                logger.error(f"加载文件失败 {file_path}: {e}")
        
        logger.info(f"共成功加载 {len(self.dataframes)} 个文件")
        return len(self.dataframes) > 0
    
    def analyze_and_merge(self) -> bool:
        """分析字段并合并数据"""
        if len(self.dataframes) < 2:
            logger.error("至少需要2个文件才能进行合并")
            return False
        
        logger.info("开始分析字段相似性...")
        
        # 收集所有字段
        field_groups = [list(df.columns) for df in self.dataframes]
        
        # AI分析字段相似性
        field_mapping = self.field_matcher.analyze_field_similarity(field_groups)
        
        # 执行合并
        logger.info("开始合并数据...")
        merged_df = self._merge_dataframes(field_mapping)
        
        if merged_df is not None:
            self.merged_df = merged_df
            self._generate_merge_report(field_mapping)
            logger.info(f"合并完成！最终数据: {len(merged_df)}行, {len(merged_df.columns)}列")
            return True
        
        return False
    
    def _merge_dataframes(self, field_mapping: Dict[str, List[str]]) -> Optional[pd.DataFrame]:
        """根据字段映射合并数据框"""
        try:
            # 创建标准化的列名映射
            standard_columns = {}
            for standard_name, similar_fields in field_mapping.items():
                for field in similar_fields:
                    standard_columns[field] = standard_name
            
            # 标准化每个数据框的列名
            standardized_dfs = []
            for i, df in enumerate(self.dataframes):
                df_copy = df.copy()
                
                # 重命名列
                rename_map = {}
                for col in df_copy.columns:
                    if col in standard_columns:
                        rename_map[col] = standard_columns[col]
                
                df_copy = df_copy.rename(columns=rename_map)
                
                # 添加数据源标识
                df_copy['_source_file'] = self.file_info[i]['file_name']
                df_copy['_source_index'] = i
                
                standardized_dfs.append(df_copy)
            
            # 合并所有数据框
            merged_df = pd.concat(standardized_dfs, ignore_index=True, sort=False)
            
            # 数据清理和去重
            merged_df = self._clean_merged_data(merged_df)
            
            return merged_df
            
        except Exception as e:
            logger.error(f"合并数据框失败: {e}")
            return None
    
    def _clean_merged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理合并后的数据"""
        logger.info("开始数据清理...")
        
        original_rows = len(df)
        
        # 移除完全重复的行（除了源文件信息）
        data_columns = [col for col in df.columns if not col.startswith('_source')]
        df_deduplicated = df.drop_duplicates(subset=data_columns, keep='first')
        
        removed_duplicates = original_rows - len(df_deduplicated)
        if removed_duplicates > 0:
            logger.info(f"移除重复数据: {removed_duplicates}行")
        
        return df_deduplicated
    
    def _generate_merge_report(self, field_mapping: Dict[str, List[str]]):
        """生成合并报告"""
        self.merge_report = {
            'merge_time': datetime.now().isoformat(),
            'source_files': self.file_info,
            'field_mapping': field_mapping,
            'merged_stats': {
                'total_rows': len(self.merged_df),
                'total_columns': len(self.merged_df.columns),
                'data_columns': len([col for col in self.merged_df.columns if not col.startswith('_source')])
            },
            'column_info': {
                col: {
                    'non_null_count': int(self.merged_df[col].count()),
                    'null_count': int(self.merged_df[col].isnull().sum()),
                    'data_type': str(self.merged_df[col].dtype)
                }
                for col in self.merged_df.columns
            }
        }
    
    def save_results(self, output_dir: str = "./output") -> Dict[str, str]:
        """保存合并结果"""
        if self.merged_df is None:
            logger.error("没有合并结果可保存")
            return {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存合并后的Excel文件
        excel_path = os.path.join(output_dir, f"merged_data_{timestamp}.xlsx")
        self.merged_df.to_excel(excel_path, index=False)
        
        # 保存合并报告
        report_path = os.path.join(output_dir, f"merge_report_{timestamp}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.merge_report, f, ensure_ascii=False, indent=2)
        
        # 保存字段映射信息
        mapping_path = os.path.join(output_dir, f"field_mapping_{timestamp}.json")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.merge_report['field_mapping'], f, ensure_ascii=False, indent=2)
        
        result_files = {
            'merged_excel': excel_path,
            'merge_report': report_path,
            'field_mapping': mapping_path
        }
        
        logger.info(f"结果已保存到: {output_dir}")
        for file_type, file_path in result_files.items():
            logger.info(f"  {file_type}: {file_path}")
        
        return result_files
    
    def print_summary(self):
        """打印合并摘要"""
        if not self.merge_report:
            logger.warning("没有合并报告可显示")
            return
        
        print("\n" + "="*60)
        print("Excel多表合并摘要报告")
        print("="*60)
        
        print(f"\n📁 源文件信息:")
        for i, file_info in enumerate(self.file_info, 1):
            print(f"  {i}. {file_info['file_name']} ({file_info['rows']}行, {file_info['column_count']}列)")
        
        print(f"\n🔗 字段映射:")
        for standard_name, similar_fields in self.merge_report['field_mapping'].items():
            if len(similar_fields) > 1:
                print(f"  {standard_name} ← {', '.join(similar_fields)}")
        
        stats = self.merge_report['merged_stats']
        print(f"\n📊 合并结果:")
        print(f"  总行数: {stats['total_rows']}")
        print(f"  总列数: {stats['total_columns']}")
        print(f"  数据列数: {stats['data_columns']}")
        
        print("\n✅ 合并完成！")
        print("="*60)

def create_sample_excel_files(output_dir: str = "./sample_data"):
    """创建示例Excel文件用于测试"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 示例数据1 - 销售报表A
    data1 = {
        '订单ID': ['A001', 'A002', 'A003', 'A004'],
        '客户名称': ['北京科技公司', '上海贸易公司', '深圳制造公司', '广州服务公司'],
        '产品名称': ['笔记本电脑', '台式机', '服务器', '打印机'],
        '销售金额': [8500, 6200, 15000, 1200],
        '销售日期': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18']
    }
    
    # 示例数据2 - 销售报表B（字段名不同）
    data2 = {
        'ID_订单': ['B001', 'B002', 'B003'],
        '客户全称': ['天津电子公司', '重庆软件公司', '成都网络公司'],
        '商品名称': ['路由器', '交换机', '防火墙'],
        '成交金额': [3200, 4500, 8800],
        '成交时间': ['2024-01-19', '2024-01-20', '2024-01-21']
    }
    
    # 示例数据3 - 销售报表C（更多变化）
    data3 = {
        '销售单号': ['C001', 'C002'],
        '公司名称': ['杭州创新公司', '南京技术公司'],
        '产品': ['云服务器', '数据库软件'],
        '金额': [12000, 9500],
        '日期': ['2024-01-22', '2024-01-23']
    }
    
    # 保存为Excel文件
    pd.DataFrame(data1).to_excel(os.path.join(output_dir, '销售报表_分公司A.xlsx'), index=False)
    pd.DataFrame(data2).to_excel(os.path.join(output_dir, '销售报表_分公司B.xlsx'), index=False)
    pd.DataFrame(data3).to_excel(os.path.join(output_dir, '销售报表_分公司C.xlsx'), index=False)
    
    logger.info(f"示例文件已创建在: {output_dir}")
    return [
        os.path.join(output_dir, '销售报表_分公司A.xlsx'),
        os.path.join(output_dir, '销售报表_分公司B.xlsx'),
        os.path.join(output_dir, '销售报表_分公司C.xlsx')
    ]

def main():
    """主函数 - 演示程序使用"""
    print("Excel多表合并工具 - AI智能匹配表头")
    print("使用DeepSeek LLM识别语义相同但表述不同的字段\n")
    
    # 创建示例文件
    print("1. 创建示例Excel文件...")
    sample_files = create_sample_excel_files()
    
    # 初始化合并器（请替换为实际的DeepSeek API Key）
    print("2. 初始化AI合并器...")
    merger = ExcelMerger(deepseek_api_key="sk-your-deepseek-api-key")
    
    # 加载文件
    print("3. 加载Excel文件...")
    if not merger.load_excel_files(sample_files):
        print("❌ 文件加载失败")
        return
    
    # 分析并合并
    print("4. AI分析字段并合并数据...")
    if not merger.analyze_and_merge():
        print("❌ 合并失败")
        return
    
    # 保存结果
    print("5. 保存合并结果...")
    result_files = merger.save_results()
    
    # 显示摘要
    merger.print_summary()
    
    print(f"\n🎉 程序执行完成！")
    print(f"📁 结果文件:")
    for file_type, file_path in result_files.items():
        print(f"  {file_type}: {file_path}")

if __name__ == "__main__":
    main()