#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据收集器模块
负责收集、存储和管理模型测试数据
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import csv


class DataCollector:
    """数据收集器主类"""
    
    def __init__(self, db_path: str = "test_results.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # 创建测试会话表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                status TEXT NOT NULL,
                total_images INTEGER,
                success_count INTEGER,
                error_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 检查并添加缺失的列
        cursor.execute("PRAGMA table_info(test_sessions)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'avg_inference_time' not in columns:
            cursor.execute("ALTER TABLE test_sessions ADD COLUMN avg_inference_time REAL")
        if 'avg_fps' not in columns:
            cursor.execute("ALTER TABLE test_sessions ADD COLUMN avg_fps REAL")
        if 'avg_accuracy' not in columns:
            cursor.execute("ALTER TABLE test_sessions ADD COLUMN avg_accuracy REAL")
        if 'avg_precision' not in columns:
            cursor.execute("ALTER TABLE test_sessions ADD COLUMN avg_precision REAL")
        if 'avg_recall' not in columns:
            cursor.execute("ALTER TABLE test_sessions ADD COLUMN avg_recall REAL")
        if 'avg_f1_score' not in columns:
            cursor.execute("ALTER TABLE test_sessions ADD COLUMN avg_f1_score REAL")
        if 'avg_memory_usage' not in columns:
            cursor.execute("ALTER TABLE test_sessions ADD COLUMN avg_memory_usage REAL")
        
        # 创建测试指标表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                image_path TEXT,
                inference_time REAL,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                memory_usage REAL,
                detection_count INTEGER,
                confidence_scores TEXT,
                bbox_data TEXT,
                FOREIGN KEY (session_id) REFERENCES test_sessions (id)
            )
        """)
        
        # 创建模型对比表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                comparison_name TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                models_compared TEXT NOT NULL,
                best_model TEXT,
                performance_summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建性能基准表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                benchmark_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                test_date TIMESTAMP NOT NULL,
                notes TEXT
            )
        """)
        
        self.conn.commit()
    
    def save_test_session(self, session_data: Dict) -> int:
        """保存测试会话"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO test_sessions 
            (session_name, model_name, dataset_name, start_time, end_time, status,
             total_images, success_count, error_count, avg_inference_time, avg_fps,
             avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_memory_usage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_data.get('session_name', ''),
            session_data.get('model_name', ''),
            session_data.get('dataset_name', ''),
            session_data.get('start_time', datetime.now()),
            session_data.get('end_time'),
            session_data.get('status', 'unknown'),
            session_data.get('total_images', 0),
            session_data.get('success_count', 0),
            session_data.get('error_count', 0),
            session_data.get('avg_inference_time', 0.0),
            session_data.get('avg_fps', 0.0),
            session_data.get('avg_accuracy', 0.0),
            session_data.get('avg_precision', 0.0),
            session_data.get('avg_recall', 0.0),
            session_data.get('avg_f1_score', 0.0),
            session_data.get('avg_memory_usage', 0.0)
        ))
        
        session_id = cursor.lastrowid
        self.conn.commit()
        return session_id
    
    def save_test_metrics(self, session_id: int, metrics: Dict):
        """保存测试指标"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO test_metrics 
            (session_id, timestamp, image_path, inference_time, accuracy, precision,
             recall, f1_score, memory_usage, detection_count, confidence_scores, bbox_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            metrics.get('timestamp', datetime.now()),
            metrics.get('image_path', ''),
            metrics.get('inference_time', 0.0),
            metrics.get('accuracy', 0.0),
            metrics.get('precision', 0.0),
            metrics.get('recall', 0.0),
            metrics.get('f1_score', 0.0),
            metrics.get('memory_usage', 0.0),
            metrics.get('detection_count', 0),
            json.dumps(metrics.get('confidence_scores', [])),
            json.dumps(metrics.get('bbox_data', []))
        ))
        
        self.conn.commit()
    
    def get_session_history(self, limit: int = 50) -> List[Dict]:
        """获取会话历史"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM test_sessions 
            ORDER BY start_time DESC 
            LIMIT ?
        """, (limit,))
        
        columns = [description[0] for description in cursor.description]
        sessions = []
        for row in cursor.fetchall():
            session = dict(zip(columns, row))
            sessions.append(session)
        
        return sessions
    
    def get_session_metrics(self, session_id: int) -> List[Dict]:
        """获取会话的详细指标"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM test_metrics 
            WHERE session_id = ? 
            ORDER BY timestamp
        """, (session_id,))
        
        columns = [description[0] for description in cursor.description]
        metrics = []
        for row in cursor.fetchall():
            metric = dict(zip(columns, row))
            # 解析JSON字段
            if metric.get('confidence_scores'):
                try:
                    metric['confidence_scores'] = json.loads(metric['confidence_scores'])
                except:
                    metric['confidence_scores'] = []
            if metric.get('bbox_data'):
                try:
                    metric['bbox_data'] = json.loads(metric['bbox_data'])
                except:
                    metric['bbox_data'] = []
            metrics.append(metric)
        
        return metrics
    
    def get_model_performance_summary(self, model_name: str, days: int = 30) -> Dict:
        """获取模型性能摘要"""
        cursor = self.conn.cursor()
        start_date = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_tests,
                AVG(avg_inference_time) as avg_inference_time,
                AVG(avg_fps) as avg_fps,
                AVG(avg_accuracy) as avg_accuracy,
                AVG(avg_precision) as avg_precision,
                AVG(avg_recall) as avg_recall,
                AVG(avg_f1_score) as avg_f1_score,
                AVG(avg_memory_usage) as avg_memory_usage
            FROM test_sessions 
            WHERE model_name = ? AND start_time >= ? AND status = 'completed'
        """, (model_name, start_date))
        
        result = cursor.fetchone()
        if result:
            return {
                'model_name': model_name,
                'total_tests': result[0],
                'avg_inference_time': result[1] or 0,
                'avg_fps': result[2] or 0,
                'avg_accuracy': result[3] or 0,
                'avg_precision': result[4] or 0,
                'avg_recall': result[5] or 0,
                'avg_f1_score': result[6] or 0,
                'avg_memory_usage': result[7] or 0
            }
        return {}
    
    def compare_models(self, model_names: List[str], dataset_name: str = None) -> Dict:
        """对比模型性能"""
        cursor = self.conn.cursor()
        
        # 构建查询条件
        where_clause = "model_name IN ({}) AND status = 'completed'".format(
            ','.join(['?' for _ in model_names])
        )
        params = model_names.copy()
        
        if dataset_name:
            where_clause += " AND dataset_name = ?"
            params.append(dataset_name)
        
        cursor.execute(f"""
            SELECT 
                model_name,
                COUNT(*) as test_count,
                AVG(avg_inference_time) as avg_inference_time,
                AVG(avg_fps) as avg_fps,
                AVG(avg_accuracy) as avg_accuracy,
                AVG(avg_precision) as avg_precision,
                AVG(avg_recall) as avg_recall,
                AVG(avg_f1_score) as avg_f1_score,
                AVG(avg_memory_usage) as avg_memory_usage
            FROM test_sessions 
            WHERE {where_clause}
            GROUP BY model_name
            ORDER BY avg_accuracy DESC
        """, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'model_name': row[0],
                'test_count': row[1],
                'avg_inference_time': row[2] or 0,
                'avg_fps': row[3] or 0,
                'avg_accuracy': row[4] or 0,
                'avg_precision': row[5] or 0,
                'avg_recall': row[6] or 0,
                'avg_f1_score': row[7] or 0,
                'avg_memory_usage': row[8] or 0
            })
        
        # 找出最佳模型
        best_model = None
        if results:
            best_model = max(results, key=lambda x: x['avg_accuracy'])
        
        return {
            'models': results,
            'best_model': best_model,
            'comparison_date': datetime.now(),
            'dataset_name': dataset_name
        }
    
    def export_session_data(self, session_id: int, export_path: str) -> bool:
        """导出会话数据"""
        try:
            # 获取会话信息
            session = self.get_session_by_id(session_id)
            if not session:
                return False
            
            # 获取详细指标
            metrics = self.get_session_metrics(session_id)
            
            # 创建导出目录
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # 导出会话信息
            session_file = export_dir / f"session_{session_id}_info.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session, f, indent=2, ensure_ascii=False, default=str)
            
            # 导出指标数据为CSV
            metrics_file = export_dir / f"session_{session_id}_metrics.csv"
            if metrics:
                df = pd.DataFrame(metrics)
                df.to_csv(metrics_file, index=False, encoding='utf-8')
            
            # 导出性能摘要
            summary_file = export_dir / f"session_{session_id}_summary.json"
            summary = {
                'session_info': session,
                'total_metrics': len(metrics),
                'avg_inference_time': np.mean([m.get('inference_time', 0) for m in metrics]) if metrics else 0,
                'avg_accuracy': np.mean([m.get('accuracy', 0) for m in metrics]) if metrics else 0,
                'avg_fps': np.mean([1000/m.get('inference_time', 1) for m in metrics if m.get('inference_time', 0) > 0]) if metrics else 0
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            return True
            
        except Exception as e:
            print(f"导出数据失败: {e}")
            return False
    
    def get_session_by_id(self, session_id: int) -> Optional[Dict]:
        """根据ID获取会话信息"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM test_sessions WHERE id = ?", (session_id,))
        
        columns = [description[0] for description in cursor.description]
        row = cursor.fetchone()
        
        if row:
            return dict(zip(columns, row))
        return None
    
    def save_model_comparison(self, comparison_data: Dict) -> int:
        """保存模型对比结果"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO model_comparisons 
            (comparison_name, dataset_name, models_compared, best_model, performance_summary)
            VALUES (?, ?, ?, ?, ?)
        """, (
            comparison_data.get('comparison_name', ''),
            comparison_data.get('dataset_name', ''),
            json.dumps(comparison_data.get('models_compared', [])),
            comparison_data.get('best_model', ''),
            json.dumps(comparison_data.get('performance_summary', {}))
        ))
        
        comparison_id = cursor.lastrowid
        self.conn.commit()
        return comparison_id
    
    def get_model_comparisons(self, limit: int = 20) -> List[Dict]:
        """获取模型对比历史"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM model_comparisons 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))
        
        columns = [description[0] for description in cursor.description]
        comparisons = []
        for row in cursor.fetchall():
            comparison = dict(zip(columns, row))
            # 解析JSON字段
            if comparison.get('models_compared'):
                try:
                    comparison['models_compared'] = json.loads(comparison['models_compared'])
                except:
                    comparison['models_compared'] = []
            if comparison.get('performance_summary'):
                try:
                    comparison['performance_summary'] = json.loads(comparison['performance_summary'])
                except:
                    comparison['performance_summary'] = {}
            comparisons.append(comparison)
        
        return comparisons
    
    def save_performance_benchmark(self, benchmark_data: Dict):
        """保存性能基准"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO performance_benchmarks 
            (model_name, dataset_name, benchmark_type, metric_name, metric_value, test_date, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            benchmark_data.get('model_name', ''),
            benchmark_data.get('dataset_name', ''),
            benchmark_data.get('benchmark_type', ''),
            benchmark_data.get('metric_name', ''),
            benchmark_data.get('metric_value', 0.0),
            benchmark_data.get('test_date', datetime.now()),
            benchmark_data.get('notes', '')
        ))
        
        self.conn.commit()
    
    def get_performance_benchmarks(self, model_name: str = None, dataset_name: str = None) -> List[Dict]:
        """获取性能基准"""
        cursor = self.conn.cursor()
        
        where_clause = "1=1"
        params = []
        
        if model_name:
            where_clause += " AND model_name = ?"
            params.append(model_name)
        
        if dataset_name:
            where_clause += " AND dataset_name = ?"
            params.append(dataset_name)
        
        cursor.execute(f"""
            SELECT * FROM performance_benchmarks 
            WHERE {where_clause}
            ORDER BY test_date DESC
        """, params)
        
        columns = [description[0] for description in cursor.description]
        benchmarks = []
        for row in cursor.fetchall():
            benchmark = dict(zip(columns, row))
            benchmarks.append(benchmark)
        
        return benchmarks
    
    def cleanup_old_data(self, days: int = 90):
        """清理旧数据"""
        cursor = self.conn.cursor()
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 删除旧的测试会话
        cursor.execute("""
            DELETE FROM test_sessions 
            WHERE start_time < ?
        """, (cutoff_date,))
        
        # 删除相关的指标数据
        cursor.execute("""
            DELETE FROM test_metrics 
            WHERE session_id NOT IN (
                SELECT id FROM test_sessions
            )
        """)
        
        deleted_sessions = cursor.rowcount
        self.conn.commit()
        
        return deleted_sessions
    
    def get_database_stats(self) -> Dict:
        """获取数据库统计信息"""
        cursor = self.conn.cursor()
        
        # 统计会话数量
        cursor.execute("SELECT COUNT(*) FROM test_sessions")
        total_sessions = cursor.fetchone()[0]
        
        # 统计指标数量
        cursor.execute("SELECT COUNT(*) FROM test_metrics")
        total_metrics = cursor.fetchone()[0]
        
        # 统计模型数量
        cursor.execute("SELECT COUNT(DISTINCT model_name) FROM test_sessions")
        unique_models = cursor.fetchone()[0]
        
        # 统计数据集数量
        cursor.execute("SELECT COUNT(DISTINCT dataset_name) FROM test_sessions")
        unique_datasets = cursor.fetchone()[0]
        
        # 获取最新测试时间
        cursor.execute("SELECT MAX(start_time) FROM test_sessions")
        latest_test = cursor.fetchone()[0]
        
        return {
            'total_sessions': total_sessions,
            'total_metrics': total_metrics,
            'unique_models': unique_models,
            'unique_datasets': unique_datasets,
            'latest_test': latest_test,
            'database_size': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        }
    
    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'conn'):
            self.conn.close()


if __name__ == "__main__":
    # 测试代码
    collector = DataCollector()
    
    # 测试基本功能
    stats = collector.get_database_stats()
    print("数据库统计:", stats)
    
    collector.close()
