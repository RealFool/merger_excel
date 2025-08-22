#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel多表合并工具 - 增强版
使用DeepSeek LLM + 模糊匹配双重策略识别语义相同但表述不同的字段

新增功能：
1. 配置文件支持
2. 模糊匹配备选方案
3. 批量处理模式
4. 更详细的合并报告
5. 错误恢复机制
6. 字段映射缓存

作者：AI Assistant
日期：2025-01-29
"""

import os
import json
import pandas as pd
import requests
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
import hashlib
from collections import defaultdict
import re
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz, process
import argparse
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('excel_merger_enhanced.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config_excel_merger.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"配置文件加载成功: {self.config_path}")
                return config
            else:
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                return self.get_default_config()
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}，使用默认配置")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "deepseek_api": {
                "api_key": "sk-your-deepseek-api-key",
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "temperature": 0.1,
                "max_tokens": 2000,
                "timeout": 30
            },
            "merge_settings": {
                "remove_duplicates": True,
                "add_source_info": True,
                "case_sensitive": False,
                "auto_detect_encoding": True
            },
            "output_settings": {
                "output_dir": "./output",
                "save_report": True,
                "save_field_mapping": True,
                "excel_format": "xlsx"
            },
            "field_matching": {
                "similarity_threshold": 0.8,
                "use_ai_analysis": True,
                "fallback_to_fuzzy_match": True,
                "common_field_patterns": {
                    "order_id": ["订单ID", "订单编号", "ID_订单", "销售单号"],
                    "customer_name": ["客户名称", "客户名", "客户全称", "公司名称"],
                    "product_name": ["产品名称", "商品名称", "产品", "商品"],
                    "amount": ["金额", "销售金额", "成交金额", "总金额"],
                    "date": ["日期", "销售日期", "成交时间", "时间"]
                }
            }
        }
    
    def get(self, key_path: str, default=None):
        """获取配置值，支持点号分隔的路径"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

class DeepSeekAPIEnhanced:
    """增强版DeepSeek API调用类"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.api_key = self.config.get('deepseek_api.api_key')
        self.base_url = self.config.get('deepseek_api.base_url')
        self.model = self.config.get('deepseek_api.model')
        self.temperature = self.config.get('deepseek_api.temperature')
        self.max_tokens = self.config.get('deepseek_api.max_tokens')
        self.timeout = self.config.get('deepseek_api.timeout')
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # API调用统计
        self.api_calls = 0
        self.api_errors = 0
    
    def call_api(self, messages: List[Dict]) -> str:
        """调用DeepSeek API"""
        try:
            self.api_calls += 1
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                self.api_errors += 1
                logger.error(f"API调用失败: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            self.api_errors += 1
            logger.error(f"API调用异常: {e}")
            return ""
    
    def get_stats(self) -> Dict:
        """获取API调用统计"""
        return {
            'total_calls': self.api_calls,
            'errors': self.api_errors,
            'success_rate': (self.api_calls - self.api_errors) / max(self.api_calls, 1)
        }

class FuzzyFieldMatcher:
    """模糊字段匹配器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.similarity_threshold = self.config.get('field_matching.similarity_threshold', 0.8)
        self.common_patterns = self.config.get('field_matching.common_field_patterns', {})
    
    def fuzzy_match_fields(self, field_groups: List[List[str]]) -> Dict[str, List[str]]:
        """使用模糊匹配算法匹配字段"""
        all_fields = []
        for group in field_groups:
            all_fields.extend(group)
        
        unique_fields = list(set(all_fields))
        
        if len(unique_fields) <= 1:
            return {unique_fields[0]: unique_fields} if unique_fields else {}
        
        # 首先尝试使用预定义模式匹配
        pattern_mapping = self._match_with_patterns(unique_fields)
        
        # 对未匹配的字段使用模糊匹配
        unmatched_fields = [f for f in unique_fields if not any(f in group for group in pattern_mapping.values())]
        fuzzy_mapping = self._fuzzy_match_remaining(unmatched_fields)
        
        # 合并结果
        final_mapping = {**pattern_mapping, **fuzzy_mapping}
        
        logger.info(f"模糊匹配结果: {final_mapping}")
        return final_mapping
    
    def _match_with_patterns(self, fields: List[str]) -> Dict[str, List[str]]:
        """使用预定义模式匹配字段"""
        mapping = {}
        matched_fields = set()
        
        for standard_name, patterns in self.common_patterns.items():
            matched_group = []
            
            for field in fields:
                if field in matched_fields:
                    continue
                
                # 检查是否匹配任何模式
                for pattern in patterns:
                    if self._is_similar(field, pattern):
                        matched_group.append(field)
                        matched_fields.add(field)
                        break
            
            if matched_group:
                mapping[standard_name] = matched_group
        
        return mapping
    
    def _fuzzy_match_remaining(self, fields: List[str]) -> Dict[str, List[str]]:
        """对剩余字段进行模糊匹配"""
        if not fields:
            return {}
        
        mapping = {}
        processed = set()
        
        for i, field1 in enumerate(fields):
            if field1 in processed:
                continue
            
            similar_group = [field1]
            processed.add(field1)
            
            for j, field2 in enumerate(fields[i+1:], i+1):
                if field2 in processed:
                    continue
                
                if self._is_similar(field1, field2):
                    similar_group.append(field2)
                    processed.add(field2)
            
            # 使用最短的字段名作为标准名
            standard_name = min(similar_group, key=len)
            mapping[standard_name] = similar_group
        
        return mapping
    
    def _is_similar(self, field1: str, field2: str) -> bool:
        """判断两个字段是否相似"""
        # 标准化字段名（去除空格、下划线，转小写）
        norm1 = re.sub(r'[\s_-]', '', field1.lower())
        norm2 = re.sub(r'[\s_-]', '', field2.lower())
        
        # 完全匹配
        if norm1 == norm2:
            return True
        
        # 模糊匹配
        similarity = fuzz.ratio(norm1, norm2) / 100.0
        return similarity >= self.similarity_threshold

class EnhancedFieldMatcher:
    """增强版字段匹配器 - 结合AI和模糊匹配"""
    
    def __init__(self, deepseek_api: DeepSeekAPIEnhanced, config_manager: ConfigManager):
        self.api = deepseek_api
        self.config = config_manager
        self.fuzzy_matcher = FuzzyFieldMatcher(config_manager)
        self.field_mapping_cache = {}
        self.use_ai = self.config.get('field_matching.use_ai_analysis', True)
        self.fallback_fuzzy = self.config.get('field_matching.fallback_to_fuzzy_match', True)
    
    def analyze_field_similarity(self, field_groups: List[List[str]]) -> Dict[str, List[str]]:
        """分析字段相似性，优先使用AI，失败时回退到模糊匹配"""
        
        # 生成缓存键
        cache_key = self._generate_cache_key(field_groups)
        if cache_key in self.field_mapping_cache:
            logger.info("使用缓存的字段映射结果")
            return self.field_mapping_cache[cache_key]
        
        # 构建所有字段列表
        all_fields = []
        for group in field_groups:
            all_fields.extend(group)
        
        unique_fields = list(set(all_fields))
        
        if len(unique_fields) <= 1:
            result = {unique_fields[0]: unique_fields} if unique_fields else {}
            self.field_mapping_cache[cache_key] = result
            return result
        
        # 尝试AI分析
        if self.use_ai:
            logger.info("使用AI分析字段相似性...")
            ai_result = self._ai_analyze_fields(unique_fields)
            
            if ai_result:
                logger.info("AI分析成功")
                self.field_mapping_cache[cache_key] = ai_result
                return ai_result
            else:
                logger.warning("AI分析失败")
        
        # 回退到模糊匹配
        if self.fallback_fuzzy:
            logger.info("使用模糊匹配作为备选方案...")
            fuzzy_result = self.fuzzy_matcher.fuzzy_match_fields(field_groups)
            self.field_mapping_cache[cache_key] = fuzzy_result
            return fuzzy_result
        
        # 最后的备选方案：每个字段自成一组
        logger.warning("所有匹配方法都失败，使用默认映射")
        default_result = {field: [field] for field in unique_fields}
        self.field_mapping_cache[cache_key] = default_result
        return default_result
    
    def _generate_cache_key(self, field_groups: List[List[str]]) -> str:
        """生成字段组的缓存键"""
        all_fields = sorted(set(field for group in field_groups for field in group))
        return hashlib.md5('|'.join(all_fields).encode()).hexdigest()
    
    def _ai_analyze_fields(self, fields: List[str]) -> Optional[Dict[str, List[str]]]:
        """使用AI分析字段"""
        try:
            prompt = self._build_field_analysis_prompt(fields)
            
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
            
            response = self.api.call_api(messages)
            
            if response:
                return self._parse_field_mapping_response(response, fields)
            
        except Exception as e:
            logger.error(f"AI字段分析异常: {e}")
        
        return None
    
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
5. 相似度较低的字段应该分开归组

请以JSON格式返回结果，格式如下：
{{
  "group_1": ["字段1", "字段2", "字段3"],
  "group_2": ["字段4", "字段5"],
  "group_3": ["字段6"]
}}

其中每个group代表一组语义相同的字段，group的命名请使用最通用的字段名。
确保每个输入字段都被分配到某个组中。
"""
        return prompt
    
    def _parse_field_mapping_response(self, response: str, original_fields: List[str]) -> Optional[Dict[str, List[str]]]:
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
                    if isinstance(group_fields, list):
                        mapped_fields.update(group_fields)
                
                # 添加未映射的字段
                for field in original_fields:
                    if field not in mapped_fields:
                        mapping[field] = [field]
                
                return mapping
        except Exception as e:
            logger.warning(f"解析AI响应失败: {e}")
        
        return None

class ExcelMergerEnhanced:
    """增强版Excel合并器"""
    
    def __init__(self, config_path: str = "config_excel_merger.json"):
        self.config_manager = ConfigManager(config_path)
        self.api = DeepSeekAPIEnhanced(self.config_manager)
        self.field_matcher = EnhancedFieldMatcher(self.api, self.config_manager)
        
        self.dataframes = []
        self.file_info = []
        self.merged_df = None
        self.merge_report = {}
        
        # 统计信息
        self.stats = {
            'start_time': None,
            'end_time': None,
            'processing_time': None,
            'files_processed': 0,
            'files_failed': 0,
            'total_rows_input': 0,
            'total_rows_output': 0,
            'duplicates_removed': 0
        }
    
    def load_excel_files(self, file_paths: List[str]) -> bool:
        """加载多个Excel文件"""
        self.stats['start_time'] = datetime.now()
        logger.info(f"开始加载 {len(file_paths)} 个Excel文件")
        
        self.dataframes = []
        self.file_info = []
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"文件不存在: {file_path}")
                    self.stats['files_failed'] += 1
                    continue
                
                # 读取Excel文件
                df = pd.read_excel(file_path)
                
                if df.empty:
                    logger.warning(f"文件为空: {file_path}")
                    self.stats['files_failed'] += 1
                    continue
                
                self.dataframes.append(df)
                self.file_info.append({
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'rows': len(df),
                    'columns': list(df.columns),
                    'column_count': len(df.columns),
                    'file_size': os.path.getsize(file_path),
                    'load_time': datetime.now().isoformat()
                })
                
                self.stats['files_processed'] += 1
                self.stats['total_rows_input'] += len(df)
                
                logger.info(f"成功加载: {os.path.basename(file_path)} ({len(df)}行, {len(df.columns)}列)")
                
            except Exception as e:
                logger.error(f"加载文件失败 {file_path}: {e}")
                self.stats['files_failed'] += 1
        
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
        
        # 分析字段相似性
        field_mapping = self.field_matcher.analyze_field_similarity(field_groups)
        
        # 执行合并
        logger.info("开始合并数据...")
        merged_df = self._merge_dataframes(field_mapping)
        
        if merged_df is not None:
            self.merged_df = merged_df
            self.stats['total_rows_output'] = len(merged_df)
            self.stats['end_time'] = datetime.now()
            self.stats['processing_time'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
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
                if self.config_manager.get('merge_settings.add_source_info', True):
                    df_copy['_source_file'] = self.file_info[i]['file_name']
                    df_copy['_source_index'] = i
                    df_copy['_row_index'] = df_copy.index
                
                standardized_dfs.append(df_copy)
            
            # 合并所有数据框
            merged_df = pd.concat(standardized_dfs, ignore_index=True, sort=False)
            
            # 数据清理
            if self.config_manager.get('merge_settings.remove_duplicates', True):
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
        data_columns = [col for col in df.columns if not col.startswith('_source') and col != '_row_index']
        if data_columns:
            df_deduplicated = df.drop_duplicates(subset=data_columns, keep='first')
        else:
            df_deduplicated = df
        
        removed_duplicates = original_rows - len(df_deduplicated)
        self.stats['duplicates_removed'] = removed_duplicates
        
        if removed_duplicates > 0:
            logger.info(f"移除重复数据: {removed_duplicates}行")
        
        return df_deduplicated
    
    def _generate_merge_report(self, field_mapping: Dict[str, List[str]]):
        """生成详细的合并报告"""
        api_stats = self.api.get_stats()
        
        self.merge_report = {
            'merge_info': {
                'merge_time': datetime.now().isoformat(),
                'processing_time_seconds': self.stats['processing_time'],
                'tool_version': '2.0.0',
                'config_file': self.config_manager.config_path
            },
            'source_files': self.file_info,
            'field_mapping': field_mapping,
            'statistics': {
                'input_stats': {
                    'files_processed': self.stats['files_processed'],
                    'files_failed': self.stats['files_failed'],
                    'total_input_rows': self.stats['total_rows_input']
                },
                'output_stats': {
                    'total_output_rows': self.stats['total_rows_output'],
                    'duplicates_removed': self.stats['duplicates_removed'],
                    'data_columns': len([col for col in self.merged_df.columns if not col.startswith('_')]),
                    'total_columns': len(self.merged_df.columns)
                },
                'api_stats': api_stats
            },
            'column_analysis': {
                col: {
                    'non_null_count': int(self.merged_df[col].count()),
                    'null_count': int(self.merged_df[col].isnull().sum()),
                    'null_percentage': float(self.merged_df[col].isnull().sum() / len(self.merged_df) * 100),
                    'data_type': str(self.merged_df[col].dtype),
                    'unique_values': int(self.merged_df[col].nunique()) if not col.startswith('_') else None
                }
                for col in self.merged_df.columns
            },
            'field_mapping_details': {
                'total_unique_fields': len(set(field for group in field_mapping.values() for field in group)),
                'mapped_groups': len(field_mapping),
                'fields_with_conflicts': len([k for k, v in field_mapping.items() if len(v) > 1])
            }
        }
    
    def save_results(self, output_dir: str = None) -> Dict[str, str]:
        """保存合并结果"""
        if self.merged_df is None:
            logger.error("没有合并结果可保存")
            return {}
        
        # 使用配置中的输出目录
        if output_dir is None:
            output_dir = self.config_manager.get('output_settings.output_dir', './output')
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result_files = {}
        
        # 保存合并后的Excel文件
        excel_format = self.config_manager.get('output_settings.excel_format', 'xlsx')
        excel_path = os.path.join(output_dir, f"merged_data_{timestamp}.{excel_format}")
        
        if excel_format == 'xlsx':
            self.merged_df.to_excel(excel_path, index=False)
        else:
            self.merged_df.to_csv(excel_path, index=False, encoding='utf-8-sig')
        
        result_files['merged_data'] = excel_path
        
        # 保存合并报告
        if self.config_manager.get('output_settings.save_report', True):
            report_path = os.path.join(output_dir, f"merge_report_{timestamp}.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.merge_report, f, ensure_ascii=False, indent=2)
            result_files['merge_report'] = report_path
        
        # 保存字段映射信息
        if self.config_manager.get('output_settings.save_field_mapping', True):
            mapping_path = os.path.join(output_dir, f"field_mapping_{timestamp}.json")
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(self.merge_report['field_mapping'], f, ensure_ascii=False, indent=2)
            result_files['field_mapping'] = mapping_path
        
        logger.info(f"结果已保存到: {output_dir}")
        for file_type, file_path in result_files.items():
            logger.info(f"  {file_type}: {file_path}")
        
        return result_files
    
    def print_summary(self):
        """打印详细的合并摘要"""
        if not self.merge_report:
            logger.warning("没有合并报告可显示")
            return
        
        print("\n" + "="*80)
        print("Excel多表合并摘要报告 - 增强版")
        print("="*80)
        
        # 处理统计
        stats = self.merge_report['statistics']
        print(f"\n⏱️  处理统计:")
        print(f"  处理时间: {self.stats['processing_time']:.2f}秒")
        print(f"  成功文件: {stats['input_stats']['files_processed']}")
        print(f"  失败文件: {stats['input_stats']['files_failed']}")
        print(f"  API调用: {stats['api_stats']['total_calls']}次 (成功率: {stats['api_stats']['success_rate']:.1%})")
        
        # 源文件信息
        print(f"\n📁 源文件信息:")
        for i, file_info in enumerate(self.file_info, 1):
            size_mb = file_info['file_size'] / 1024 / 1024
            print(f"  {i}. {file_info['file_name']} ({file_info['rows']}行, {file_info['column_count']}列, {size_mb:.1f}MB)")
        
        # 字段映射
        print(f"\n🔗 字段映射:")
        mapping_details = self.merge_report['field_mapping_details']
        print(f"  原始字段数: {mapping_details['total_unique_fields']}")
        print(f"  映射组数: {mapping_details['mapped_groups']}")
        print(f"  有冲突的字段组: {mapping_details['fields_with_conflicts']}")
        
        for standard_name, similar_fields in self.merge_report['field_mapping'].items():
            if len(similar_fields) > 1:
                print(f"    {standard_name} ← {', '.join(similar_fields)}")
        
        # 合并结果
        output_stats = stats['output_stats']
        print(f"\n📊 合并结果:")
        print(f"  输入总行数: {stats['input_stats']['total_input_rows']:,}")
        print(f"  输出总行数: {output_stats['total_output_rows']:,}")
        print(f"  移除重复: {output_stats['duplicates_removed']:,}行")
        print(f"  数据列数: {output_stats['data_columns']}")
        print(f"  总列数: {output_stats['total_columns']}")
        
        # 数据质量
        print(f"\n📈 数据质量:")
        data_columns = [col for col in self.merged_df.columns if not col.startswith('_')]
        for col in data_columns[:5]:  # 只显示前5列
            col_info = self.merge_report['column_analysis'][col]
            print(f"  {col}: {col_info['non_null_count']:,}非空 ({100-col_info['null_percentage']:.1f}%)")
        
        if len(data_columns) > 5:
            print(f"  ... 还有 {len(data_columns)-5} 列")
        
        print("\n✅ 合并完成！")
        print("="*80)

def create_enhanced_sample_files(output_dir: str = "./sample_data_enhanced"):
    """创建增强版示例Excel文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 示例数据1 - 销售报表A
    data1 = {
        '订单ID': ['A001', 'A002', 'A003', 'A004', 'A005'],
        '客户名称': ['北京科技有限公司', '上海贸易集团', '深圳制造企业', '广州服务公司', '杭州创新科技'],
        '产品名称': ['笔记本电脑', '台式机', '服务器', '打印机', '路由器'],
        '销售金额': [8500, 6200, 15000, 1200, 3200],
        '销售日期': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
        '销售员': ['张三', '李四', '王五', '赵六', '钱七']
    }
    
    # 示例数据2 - 销售报表B（字段名不同）
    data2 = {
        'ID_订单': ['B001', 'B002', 'B003', 'B004'],
        '客户全称': ['天津电子科技公司', '重庆软件开发公司', '成都网络技术公司', '西安通信设备公司'],
        '商品名称': ['交换机', '防火墙', '存储设备', '监控系统'],
        '成交金额': [4500, 8800, 12000, 6500],
        '成交时间': ['2024-01-20', '2024-01-21', '2024-01-22', '2024-01-23'],
        '业务员': ['孙八', '周九', '吴十', '郑一']
    }
    
    # 示例数据3 - 销售报表C（更多变化）
    data3 = {
        '销售单号': ['C001', 'C002', 'C003'],
        '公司名称': ['南京技术有限公司', '苏州制造集团', '无锡贸易公司'],
        '产品': ['云服务器', '数据库软件', '办公软件'],
        '金额': [18000, 9500, 3500],
        '日期': ['2024-01-24', '2024-01-25', '2024-01-26'],
        '负责人': ['陈二', '林三', '黄四']
    }
    
    # 示例数据4 - 包含重复数据
    data4 = {
        'order_id': ['A001', 'D001', 'D002'],  # A001是重复数据
        'customer_name': ['北京科技有限公司', '青岛海洋公司', '烟台港口公司'],
        'product_name': ['笔记本电脑', '船舶设备', '港口机械'],
        'amount': [8500, 25000, 18000],
        'sale_date': ['2024-01-15', '2024-01-27', '2024-01-28'],
        'salesperson': ['张三', '刘五', '陈六']
    }
    
    # 保存为Excel文件
    files = [
        ('销售报表_总部.xlsx', data1),
        ('销售报表_华北分公司.xlsx', data2),
        ('销售报表_华东分公司.xlsx', data3),
        ('销售报表_华南分公司.xlsx', data4)
    ]
    
    file_paths = []
    for filename, data in files:
        file_path = os.path.join(output_dir, filename)
        pd.DataFrame(data).to_excel(file_path, index=False)
        file_paths.append(file_path)
    
    logger.info(f"增强版示例文件已创建在: {output_dir}")
    return file_paths

def main():
    """主函数 - 支持命令行参数"""
    parser = argparse.ArgumentParser(description='Excel多表合并工具 - AI智能匹配表头')
    parser.add_argument('--files', nargs='+', help='要合并的Excel文件路径')
    parser.add_argument('--config', default='config_excel_merger.json', help='配置文件路径')
    parser.add_argument('--output', help='输出目录')
    parser.add_argument('--demo', action='store_true', help='运行演示模式')
    parser.add_argument('--create-sample', action='store_true', help='创建示例文件')
    
    args = parser.parse_args()
    
    print("Excel多表合并工具 - AI智能匹配表头 (增强版)")
    print("使用DeepSeek LLM + 模糊匹配识别语义相同但表述不同的字段\n")
    
    # 创建示例文件
    if args.create_sample:
        print("创建示例Excel文件...")
        sample_files = create_enhanced_sample_files()
        print(f"示例文件已创建: {sample_files}")
        return
    
    # 演示模式
    if args.demo:
        print("1. 创建示例Excel文件...")
        sample_files = create_enhanced_sample_files()
        file_paths = sample_files
    else:
        if not args.files:
            print("错误: 请指定要合并的Excel文件路径")
            print("使用 --help 查看帮助信息")
            return
        file_paths = args.files
    
    # 初始化合并器
    print("2. 初始化AI合并器...")
    merger = ExcelMergerEnhanced(args.config)
    
    # 加载文件
    print("3. 加载Excel文件...")
    if not merger.load_excel_files(file_paths):
        print("❌ 文件加载失败")
        return
    
    # 分析并合并
    print("4. AI分析字段并合并数据...")
    if not merger.analyze_and_merge():
        print("❌ 合并失败")
        return
    
    # 保存结果
    print("5. 保存合并结果...")
    result_files = merger.save_results(args.output)
    
    # 显示摘要
    merger.print_summary()
    
    print(f"\n🎉 程序执行完成！")
    print(f"📁 结果文件:")
    for file_type, file_path in result_files.items():
        print(f"  {file_type}: {file_path}")

if __name__ == "__main__":
    main()