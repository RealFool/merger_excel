#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel多表合并工具 - 演示版
专为演示设计，无需API密钥即可运行，主要使用模糊匹配算法

功能特点：
1. 无需API密钥，开箱即用
2. 智能模糊匹配算法
3. 预定义字段模式匹配
4. 完整的合并报告
5. 支持批量处理

作者：AI Assistant
日期：2025-01-29
"""

import os
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime
import hashlib
from collections import defaultdict
import re
from fuzzywuzzy import fuzz, process
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('excel_merger_demo.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DemoFieldMatcher:
    """演示版字段匹配器 - 基于模糊匹配和预定义规则"""
    
    def __init__(self):
        self.similarity_threshold = 0.75
        self.common_patterns = {
            "订单ID": ["订单ID", "订单编号", "ID_订单", "销售单号", "order_id", "order_number", "单号"],
            "客户名称": ["客户名称", "客户名", "客户全称", "公司名称", "customer_name", "company_name", "客户"],
            "产品名称": ["产品名称", "商品名称", "产品", "商品", "product_name", "item_name", "货品"],
            "金额": ["金额", "销售金额", "成交金额", "总金额", "amount", "price", "total", "价格"],
            "日期": ["日期", "销售日期", "成交时间", "时间", "date", "time", "created_at", "订单日期"],
            "销售员": ["销售员", "业务员", "负责人", "salesperson", "sales_rep", "员工"]
        }
        
        # 统计信息
        self.match_stats = {
            'pattern_matches': 0,
            'fuzzy_matches': 0,
            'no_matches': 0
        }
    
    def analyze_field_similarity(self, field_groups: List[List[str]]) -> Dict[str, List[str]]:
        """分析字段相似性"""
        all_fields = []
        for group in field_groups:
            all_fields.extend(group)
        
        unique_fields = list(set(all_fields))
        
        if len(unique_fields) <= 1:
            return {unique_fields[0]: unique_fields} if unique_fields else {}
        
        logger.info(f"开始分析 {len(unique_fields)} 个字段的相似性...")
        
        # 首先使用预定义模式匹配
        pattern_mapping = self._match_with_patterns(unique_fields)
        
        # 对未匹配的字段使用模糊匹配
        unmatched_fields = [f for f in unique_fields if not any(f in group for group in pattern_mapping.values())]
        fuzzy_mapping = self._fuzzy_match_remaining(unmatched_fields)
        
        # 合并结果
        final_mapping = {**pattern_mapping, **fuzzy_mapping}
        
        # 记录统计信息
        self._log_match_results(final_mapping, unique_fields)
        
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
                    if self._is_similar(field, pattern, threshold=0.85):
                        matched_group.append(field)
                        matched_fields.add(field)
                        self.match_stats['pattern_matches'] += 1
                        logger.debug(f"模式匹配: {field} -> {standard_name} (模式: {pattern})")
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
                    self.match_stats['fuzzy_matches'] += 1
                    logger.debug(f"模糊匹配: {field1} <-> {field2}")
            
            # 使用最短的字段名作为标准名
            standard_name = min(similar_group, key=len)
            mapping[standard_name] = similar_group
        
        # 统计未匹配的字段
        for field in fields:
            if field in processed:
                continue
            self.match_stats['no_matches'] += 1
        
        return mapping
    
    def _is_similar(self, field1: str, field2: str, threshold: float = None) -> bool:
        """判断两个字段是否相似"""
        if threshold is None:
            threshold = self.similarity_threshold
        
        # 标准化字段名（去除空格、下划线，转小写）
        norm1 = re.sub(r'[\s_-]', '', field1.lower())
        norm2 = re.sub(r'[\s_-]', '', field2.lower())
        
        # 完全匹配
        if norm1 == norm2:
            return True
        
        # 包含关系检查
        if norm1 in norm2 or norm2 in norm1:
            return True
        
        # 模糊匹配
        similarity = fuzz.ratio(norm1, norm2) / 100.0
        return similarity >= threshold
    
    def _log_match_results(self, mapping: Dict[str, List[str]], original_fields: List[str]):
        """记录匹配结果"""
        logger.info(f"字段匹配完成:")
        logger.info(f"  原始字段数: {len(original_fields)}")
        logger.info(f"  映射组数: {len(mapping)}")
        logger.info(f"  模式匹配: {self.match_stats['pattern_matches']}")
        logger.info(f"  模糊匹配: {self.match_stats['fuzzy_matches']}")
        logger.info(f"  未匹配: {self.match_stats['no_matches']}")
        
        # 显示有冲突的字段组
        conflicts = [(k, v) for k, v in mapping.items() if len(v) > 1]
        if conflicts:
            logger.info(f"发现 {len(conflicts)} 个字段组有多个变体:")
            for standard_name, variants in conflicts:
                logger.info(f"  {standard_name}: {', '.join(variants)}")

class ExcelMergerDemo:
    """演示版Excel合并器"""
    
    def __init__(self):
        self.field_matcher = DemoFieldMatcher()
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
                
                logger.info(f"✅ {os.path.basename(file_path)} ({len(df)}行, {len(df.columns)}列)")
                logger.info(f"   字段: {', '.join(df.columns)}")
                
            except Exception as e:
                logger.error(f"❌ 加载失败 {file_path}: {e}")
                self.stats['files_failed'] += 1
        
        logger.info(f"\n📊 加载汇总: 成功 {self.stats['files_processed']} 个，失败 {self.stats['files_failed']} 个")
        return len(self.dataframes) > 0
    
    def analyze_and_merge(self) -> bool:
        """分析字段并合并数据"""
        if len(self.dataframes) < 2:
            logger.error("❌ 至少需要2个文件才能进行合并")
            return False
        
        logger.info("\n🔍 开始智能字段分析...")
        
        # 收集所有字段
        field_groups = [list(df.columns) for df in self.dataframes]
        
        # 分析字段相似性
        field_mapping = self.field_matcher.analyze_field_similarity(field_groups)
        
        # 执行合并
        logger.info("\n🔄 开始数据合并...")
        merged_df = self._merge_dataframes(field_mapping)
        
        if merged_df is not None:
            self.merged_df = merged_df
            self.stats['total_rows_output'] = len(merged_df)
            self.stats['end_time'] = datetime.now()
            self.stats['processing_time'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            self._generate_merge_report(field_mapping)
            logger.info(f"✅ 合并完成！最终数据: {len(merged_df):,}行, {len(merged_df.columns)}列")
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
                
                if rename_map:
                    df_copy = df_copy.rename(columns=rename_map)
                    logger.info(f"📝 文件 {i+1} 字段重命名: {rename_map}")
                
                # 添加数据源标识
                df_copy['_数据源'] = self.file_info[i]['file_name']
                df_copy['_源文件序号'] = i + 1
                
                standardized_dfs.append(df_copy)
            
            # 合并所有数据框
            merged_df = pd.concat(standardized_dfs, ignore_index=True, sort=False)
            
            # 数据清理
            merged_df = self._clean_merged_data(merged_df)
            
            return merged_df
            
        except Exception as e:
            logger.error(f"❌ 合并数据框失败: {e}")
            return None
    
    def _clean_merged_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理合并后的数据"""
        logger.info("🧹 开始数据清理...")
        
        original_rows = len(df)
        
        # 移除完全重复的行（除了源文件信息）
        data_columns = [col for col in df.columns if not col.startswith('_')]
        if data_columns:
            df_deduplicated = df.drop_duplicates(subset=data_columns, keep='first')
        else:
            df_deduplicated = df
        
        removed_duplicates = original_rows - len(df_deduplicated)
        self.stats['duplicates_removed'] = removed_duplicates
        
        if removed_duplicates > 0:
            logger.info(f"🗑️  移除重复数据: {removed_duplicates}行")
        else:
            logger.info("✨ 未发现重复数据")
        
        return df_deduplicated
    
    def _generate_merge_report(self, field_mapping: Dict[str, List[str]]):
        """生成详细的合并报告"""
        self.merge_report = {
            'merge_info': {
                'merge_time': datetime.now().isoformat(),
                'processing_time_seconds': round(self.stats['processing_time'], 2),
                'tool_version': 'Demo 1.0.0',
                'merge_method': 'Fuzzy Matching + Pattern Recognition'
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
                'matching_stats': self.field_matcher.match_stats
            },
            'column_analysis': {
                col: {
                    'non_null_count': int(self.merged_df[col].count()),
                    'null_count': int(self.merged_df[col].isnull().sum()),
                    'null_percentage': round(float(self.merged_df[col].isnull().sum() / len(self.merged_df) * 100), 2),
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
    
    def save_results(self, output_dir: str = "./output") -> Dict[str, str]:
        """保存合并结果"""
        if self.merged_df is None:
            logger.error("❌ 没有合并结果可保存")
            return {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result_files = {}
        
        # 保存合并后的Excel文件
        excel_path = os.path.join(output_dir, f"合并结果_{timestamp}.xlsx")
        self.merged_df.to_excel(excel_path, index=False)
        result_files['merged_data'] = excel_path
        
        # 保存合并报告
        report_path = os.path.join(output_dir, f"合并报告_{timestamp}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.merge_report, f, ensure_ascii=False, indent=2)
        result_files['merge_report'] = report_path
        
        # 保存字段映射信息
        mapping_path = os.path.join(output_dir, f"字段映射_{timestamp}.json")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(self.merge_report['field_mapping'], f, ensure_ascii=False, indent=2)
        result_files['field_mapping'] = mapping_path
        
        logger.info(f"\n💾 结果已保存到: {output_dir}")
        for file_type, file_path in result_files.items():
            logger.info(f"   📄 {os.path.basename(file_path)}")
        
        return result_files
    
    def print_summary(self):
        """打印详细的合并摘要"""
        if not self.merge_report:
            logger.warning("⚠️  没有合并报告可显示")
            return
        
        print("\n" + "="*80)
        print("🎯 Excel多表合并摘要报告 - 演示版")
        print("="*80)
        
        # 处理统计
        stats = self.merge_report['statistics']
        print(f"\n⏱️  处理统计:")
        print(f"   处理时间: {self.stats['processing_time']:.2f}秒")
        print(f"   成功文件: {stats['input_stats']['files_processed']}个")
        print(f"   失败文件: {stats['input_stats']['files_failed']}个")
        
        # 匹配统计
        match_stats = stats['matching_stats']
        print(f"\n🎯 字段匹配统计:")
        print(f"   模式匹配: {match_stats['pattern_matches']}次")
        print(f"   模糊匹配: {match_stats['fuzzy_matches']}次")
        print(f"   未匹配: {match_stats['no_matches']}次")
        
        # 源文件信息
        print(f"\n📁 源文件信息:")
        for i, file_info in enumerate(self.file_info, 1):
            size_mb = file_info['file_size'] / 1024 / 1024
            print(f"   {i}. {file_info['file_name']}")
            print(f"      📊 {file_info['rows']:,}行 × {file_info['column_count']}列 ({size_mb:.1f}MB)")
            print(f"      📝 字段: {', '.join(file_info['columns'])}")
        
        # 字段映射
        print(f"\n🔗 智能字段映射:")
        mapping_details = self.merge_report['field_mapping_details']
        print(f"   原始字段数: {mapping_details['total_unique_fields']}")
        print(f"   映射组数: {mapping_details['mapped_groups']}")
        print(f"   有变体的字段组: {mapping_details['fields_with_conflicts']}")
        
        print(f"\n   📋 具体映射关系:")
        for standard_name, similar_fields in self.merge_report['field_mapping'].items():
            if len(similar_fields) > 1:
                print(f"      🎯 {standard_name} ← [{', '.join(similar_fields)}]")
            else:
                print(f"      📌 {standard_name}")
        
        # 合并结果
        output_stats = stats['output_stats']
        print(f"\n📊 合并结果:")
        print(f"   输入总行数: {stats['input_stats']['total_input_rows']:,}")
        print(f"   输出总行数: {output_stats['total_output_rows']:,}")
        print(f"   移除重复: {output_stats['duplicates_removed']:,}行")
        print(f"   数据列数: {output_stats['data_columns']}")
        print(f"   总列数: {output_stats['total_columns']}")
        
        # 数据质量预览
        print(f"\n📈 数据质量预览:")
        data_columns = [col for col in self.merged_df.columns if not col.startswith('_')]
        for col in data_columns[:5]:  # 只显示前5列
            col_info = self.merge_report['column_analysis'][col]
            completeness = 100 - col_info['null_percentage']
            print(f"   📊 {col}: {col_info['non_null_count']:,}非空值 ({completeness:.1f}%完整度)")
        
        if len(data_columns) > 5:
            print(f"   ... 还有 {len(data_columns)-5} 列")
        
        print("\n✅ 合并完成！数据已准备就绪")
        print("="*80)

def create_demo_sample_files(output_dir: str = "./demo_sample_data"):
    """创建演示用的示例Excel文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 示例数据1 - 总部销售报表
    data1 = {
        '订单ID': ['HQ001', 'HQ002', 'HQ003', 'HQ004', 'HQ005'],
        '客户名称': ['北京科技有限公司', '上海贸易集团有限公司', '深圳制造企业', '广州服务公司', '杭州创新科技'],
        '产品名称': ['笔记本电脑', '台式机', '服务器', '打印机', '路由器'],
        '销售金额': [8500, 6200, 15000, 1200, 3200],
        '销售日期': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
        '销售员': ['张三', '李四', '王五', '赵六', '钱七']
    }
    
    # 示例数据2 - 华北分公司（字段名略有不同）
    data2 = {
        'ID_订单': ['BJ001', 'BJ002', 'BJ003', 'BJ004'],
        '客户全称': ['天津电子科技公司', '石家庄软件开发公司', '太原网络技术公司', '呼和浩特通信设备公司'],
        '商品名称': ['交换机', '防火墙', '存储设备', '监控系统'],
        '成交金额': [4500, 8800, 12000, 6500],
        '成交时间': ['2024-01-20', '2024-01-21', '2024-01-22', '2024-01-23'],
        '业务员': ['孙八', '周九', '吴十', '郑一']
    }
    
    # 示例数据3 - 华东分公司（字段名差异更大）
    data3 = {
        '销售单号': ['SH001', 'SH002', 'SH003'],
        '公司名称': ['南京技术有限公司', '苏州制造集团', '无锡贸易公司'],
        '产品': ['云服务器', '数据库软件', '办公软件'],
        '金额': [18000, 9500, 3500],
        '日期': ['2024-01-24', '2024-01-25', '2024-01-26'],
        '负责人': ['陈二', '林三', '黄四']
    }
    
    # 示例数据4 - 华南分公司（包含重复数据和英文字段）
    data4 = {
        'order_id': ['HQ001', 'GZ001', 'GZ002'],  # HQ001是重复数据
        'customer_name': ['北京科技有限公司', '深圳海洋公司', '珠海港口公司'],
        'product_name': ['笔记本电脑', '船舶设备', '港口机械'],
        'amount': [8500, 25000, 18000],
        'sale_date': ['2024-01-15', '2024-01-27', '2024-01-28'],
        'salesperson': ['张三', '刘五', '陈六']
    }
    
    # 示例数据5 - 西南分公司（更多字段变体）
    data5 = {
        '单号': ['CD001', 'CD002'],
        '客户': ['成都软件公司', '重庆制造企业'],
        '货品': ['ERP系统', '生产设备'],
        '价格': [45000, 32000],
        '订单日期': ['2024-01-29', '2024-01-30'],
        '员工': ['王七', '李八']
    }
    
    # 保存为Excel文件
    files = [
        ('销售报表_总部.xlsx', data1),
        ('销售报表_华北分公司.xlsx', data2),
        ('销售报表_华东分公司.xlsx', data3),
        ('销售报表_华南分公司.xlsx', data4),
        ('销售报表_西南分公司.xlsx', data5)
    ]
    
    file_paths = []
    for filename, data in files:
        file_path = os.path.join(output_dir, filename)
        pd.DataFrame(data).to_excel(file_path, index=False)
        file_paths.append(file_path)
    
    logger.info(f"📁 演示示例文件已创建在: {output_dir}")
    return file_paths

def main():
    """主函数 - 演示版"""
    parser = argparse.ArgumentParser(description='Excel多表合并工具 - 演示版（无需API密钥）')
    parser.add_argument('--files', nargs='+', help='要合并的Excel文件路径')
    parser.add_argument('--output', default='./demo_output', help='输出目录')
    parser.add_argument('--demo', action='store_true', help='运行完整演示')
    parser.add_argument('--create-sample', action='store_true', help='仅创建示例文件')
    
    args = parser.parse_args()
    
    print("🚀 Excel多表合并工具 - 演示版")
    print("💡 基于智能模糊匹配算法，无需API密钥即可使用")
    print("🎯 自动识别语义相同但表述不同的字段名\n")
    
    # 仅创建示例文件
    if args.create_sample:
        print("📝 创建演示示例文件...")
        sample_files = create_demo_sample_files()
        print(f"✅ 示例文件已创建: {len(sample_files)}个")
        for file_path in sample_files:
            print(f"   📄 {os.path.basename(file_path)}")
        return
    
    # 完整演示模式
    if args.demo:
        print("🎬 1. 创建演示示例文件...")
        sample_files = create_demo_sample_files()
        file_paths = sample_files
        print(f"✅ 已创建 {len(sample_files)} 个示例文件")
    else:
        if not args.files:
            print("❌ 错误: 请指定要合并的Excel文件路径")
            print("💡 使用 --demo 运行演示，或 --help 查看帮助")
            return
        file_paths = args.files
    
    # 初始化合并器
    print("\n🤖 2. 初始化智能合并器...")
    merger = ExcelMergerDemo()
    
    # 加载文件
    print("\n📂 3. 加载Excel文件...")
    if not merger.load_excel_files(file_paths):
        print("❌ 文件加载失败")
        return
    
    # 分析并合并
    print("\n🧠 4. 智能分析字段并合并数据...")
    if not merger.analyze_and_merge():
        print("❌ 合并失败")
        return
    
    # 保存结果
    print("\n💾 5. 保存合并结果...")
    result_files = merger.save_results(args.output)
    
    # 显示摘要
    merger.print_summary()
    
    print(f"\n🎉 演示完成！")
    print(f"📁 结果文件位置: {args.output}")
    for file_type, file_path in result_files.items():
        print(f"   📄 {os.path.basename(file_path)}")
    
    print(f"\n💡 提示: 您可以打开Excel文件查看合并结果")
    print(f"📊 合并报告包含详细的字段映射和统计信息")

if __name__ == "__main__":
    main()