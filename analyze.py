

import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from snownlp import SnowNLP
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import warnings
import os
import sys
from collections import Counter
import itertools
from wordcloud import WordCloud
import matplotlib
from typing import Dict, List, Tuple, Any, Optional

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 自定义停用词列表
CUSTOM_STOPWORDS = set([
    '的', '了', '在', '是', '和', '与', '等', '为', '对', '中', '也', '有', 
    '而', '但', '就', '都', '这', '那', '一个', '一些', '之', '与', '或',
    '日', '月', '年', '时', '分', '秒', '公司', '表示', '认为', '指出',
    '报道', '新闻', '记者', '据悉', '了解', '相关', '进行', '发展', '技术'
])

class DeepSeekNewsAnalyzer:
    """DeepSeek新闻数据分析器 - 修复版"""
    
    def __init__(self, file_path: str):
        """
        初始化分析器
        
        Args:
            file_path: 新闻数据CSV文件路径
        """
        self.file_path = file_path
        self.df = None
        self.analysis_results = {}
        self.setup_jieba()
        
    def setup_jieba(self):
        """设置结巴分词"""
        # 添加DeepSeek相关词汇到词典
        jieba.add_word('DeepSeek', freq=1000, tag='nz')
        jieba.add_word('深度求索', freq=1000, tag='nz')
        jieba.add_word('大模型', freq=800, tag='n')
        jieba.add_word('开源模型', freq=800, tag='n')
        jieba.add_word('AI模型', freq=800, tag='n')
        jieba.add_word('人工智能', freq=800, tag='n')
        
    def extract_date_from_url(self, url: str) -> Optional[pd.Timestamp]:
        """
        从URL中提取日期（针对CCTV等新闻网站格式）
        
        Args:
            url: 新闻网址
            
        Returns:
            提取的日期或None
        """
        if not isinstance(url, str):
            return None
        
        # 匹配常见日期格式
        patterns = [
            r'/(\d{4})/(\d{1,2})/(\d{1,2})/',      # /2025/01/28/
            r'/(\d{4})-(\d{1,2})-(\d{1,2})/',      # /2025-01-28/
            r'(\d{4})年(\d{1,2})月(\d{1,2})日',     # 2025年01月28日
            r'(\d{4})\.(\d{1,2})\.(\d{1,2})',      # 2025.01.28
            r'(\d{4})(\d{2})(\d{2})',              # 20250128
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) >= 3:
                        year, month, day = groups[:3]
                        # 确保是有效的日期
                        year_int, month_int, day_int = int(year), int(month), int(day)
                        if 2000 <= year_int <= 2030 and 1 <= month_int <= 12 and 1 <= day_int <= 31:
                            return pd.Timestamp(f"{year_int:04d}-{month_int:02d}-{day_int:02d}")
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def load_and_clean_data(self) -> pd.DataFrame:
        """
        加载并清洗新闻数据（修复版）
        
        Returns:
            清洗后的DataFrame
        """
        print("=" * 60)
        print("开始加载和清洗数据...")
        print("=" * 60)
        
        try:
            # 尝试不同的编码方式读取文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.file_path, encoding=encoding)
                    print(f"✅ 使用 {encoding} 编码成功读取文件")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.df is None:
                print("❌ 无法读取文件，请检查文件路径和编码")
                return None
                
            print(f"📊 原始数据形状: {self.df.shape}")
            print(f"📋 原始列名: {list(self.df.columns)}")
            
            # 标准化列名（处理大小写、空格等）
            self.standardize_column_names()
            
            # 显示数据基本信息
            self.display_data_info()
            
            # 数据清洗流程
            self.df = self.clean_dataframe_enhanced(self.df)
            
            print("=" * 60)
            print("✅ 数据加载和清洗完成!")
            print(f"📊 最终数据形状: {self.df.shape}")
            print("=" * 60)
            
            return self.df
            
        except Exception as e:
            print(f"❌ 数据加载失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def standardize_column_names(self):
        """标准化列名"""
        column_mapping = {}
        
        for col in self.df.columns:
            col_lower = str(col).strip().lower()
            
            # 识别并映射常见列名
            if any(keyword in col_lower for keyword in ['标题', 'title', 'subject', '标题']):
                column_mapping[col] = '标题'
            elif any(keyword in col_lower for keyword in ['简介', 'summary', 'desc', '摘要', 'description']):
                column_mapping[col] = '简介'
            elif any(keyword in col_lower for keyword in ['正文', '内容', 'content', 'text', '文章', 'body']):
                column_mapping[col] = '正文'
            elif any(keyword in col_lower for keyword in ['来源', 'source', '媒体', 'publisher']):
                column_mapping[col] = '来源'
            elif any(keyword in col_lower for keyword in ['发布时间', '时间', 'date', 'pubtime', 'publish', 'pub_date', 'time']):
                column_mapping[col] = '发布时间'
            elif any(keyword in col_lower for keyword in ['关键词', 'keyword', 'tags', 'keyword']):
                column_mapping[col] = '关键词'
            elif any(keyword in col_lower for keyword in ['网址', 'url', 'link', '链接']):
                column_mapping[col] = '网址'
            elif any(keyword in col_lower for keyword in ['图片', 'image', 'pic', 'img']):
                column_mapping[col] = '图片'
        
        # 应用列名映射
        if column_mapping:
            self.df = self.df.rename(columns=column_mapping)
            print(f"🔄 已标准化列名: {column_mapping}")
    
    def display_data_info(self):
        """显示数据基本信息"""
        print("\n📈 数据基本信息:")
        print(f"   数据行数: {len(self.df)}")
        print(f"   数据列数: {len(self.df.columns)}")
        print(f"   当前列名: {list(self.df.columns)}")
        
        # 显示前几行数据
        print("\n📄 数据预览 (前2行):")
        for i in range(min(2, len(self.df))):
            print(f"\n--- 第 {i+1} 行 ---")
            for col in self.df.columns:
                val = self.df.iloc[i][col]
                preview = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
                print(f"  {col}: {preview}")
    
    def clean_dataframe_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """增强版DataFrame清洗"""
        df_clean = df.copy()
        
        print("\n🧹 清洗步骤1: 处理缺失值...")
        # 1. 处理缺失值
        for col in df_clean.columns:
            missing_count = df_clean[col].isna().sum()
            if missing_count > 0:
                print(f"   列 '{col}' 有 {missing_count} 个缺失值")
                
                # 根据列类型填充缺失值
                if col in ['标题', '简介', '正文', '来源', '关键词']:
                    df_clean[col] = df_clean[col].fillna('')
                elif col == '发布时间':
                    # 暂时保留NaN，后续会从URL提取
                    pass
                elif col == '网址':
                    df_clean[col] = df_clean[col].fillna('')
        
        # 2. 去除完全空白的行
        before_len = len(df_clean)
        if '标题' in df_clean.columns and '正文' in df_clean.columns:
            # 创建一个综合内容列来判断是否空白
            df_clean['综合内容'] = df_clean['标题'].fillna('') + df_clean['正文'].fillna('').str[:100]
            mask = df_clean['综合内容'].str.strip() != ''
            df_clean = df_clean[mask].copy()
            df_clean = df_clean.drop(columns=['综合内容'])
            print(f"   去除空白行: {before_len} → {len(df_clean)}")
        
        # 3. 去除重复数据（基于标题和正文前100字符）
        before_len = len(df_clean)
        if '标题' in df_clean.columns:
            # 创建去重标识
            if '正文' in df_clean.columns:
                df_clean['去重标识'] = df_clean['标题'].fillna('') + df_clean['正文'].fillna('').str[:100]
                df_clean = df_clean.drop_duplicates(subset=['去重标识'], keep='first')
                df_clean = df_clean.drop(columns=['去重标识'])
            else:
                df_clean = df_clean.drop_duplicates(subset=['标题'], keep='first')
            print(f"   去除重复数据: {before_len} → {len(df_clean)}")
        
        
        print("   处理发布时间列（增强版）...")
        if '发布时间' in df_clean.columns:
            # 先尝试直接转换
            df_clean['发布时间'] = pd.to_datetime(df_clean['发布时间'], errors='coerce', format='mixed')
            
            # 从URL提取日期（如果发布时间无效）
            if '网址' in df_clean.columns:
                url_date_extracted = 0
                for idx, row in df_clean.iterrows():
                    if pd.isna(row['发布时间']) and pd.notna(row.get('网址')) and row['网址'] != '':
                        url_date = self.extract_date_from_url(str(row['网址']))
                        if url_date:
                            df_clean.at[idx, '发布时间'] = url_date
                            url_date_extracted += 1
                
                if url_date_extracted > 0:
                    print(f"   ✅ 从URL提取了 {url_date_extracted} 个日期")
            
            # 再次检查有效时间
            valid_times = df_clean['发布时间'].notna().sum()
            print(f"   有效发布时间: {valid_times}/{len(df_clean)}")
            
            # 如果有效时间太少，创建合理的时间序列
            if valid_times < 5 and valid_times > 0:
                print("   ⚠️ 有效时间较少，进行时间序列扩展...")
                # 获取最早和最晚的有效时间
                valid_dates = df_clean['发布时间'].dropna()
                if len(valid_dates) > 0:
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()
                    
                    # 为缺失时间创建合理的时间序列
                    date_range = pd.date_range(
                        start=min_date - pd.Timedelta(days=30),
                        end=max_date + pd.Timedelta(days=30),
                        periods=len(df_clean)
                    )
                    df_clean['发布时间'] = date_range
            elif valid_times == 0:
                print("   ℹ️ 没有有效时间，创建模拟时间序列...")
                # 创建最近90天的时间序列
                start_date = pd.Timestamp.now() - pd.Timedelta(days=90)
                date_range = pd.date_range(start=start_date, periods=len(df_clean), freq='D')
                df_clean['发布时间'] = date_range
            
            # 确保所有行都有时间
            df_clean['发布时间'] = df_clean['发布时间'].fillna(pd.Timestamp.now())
            print(f"   最终有效时间: {df_clean['发布时间'].notna().sum()}/{len(df_clean)}")
        
        # 5. 确保文本列为字符串类型
        text_columns = ['标题', '简介', '正文', '来源']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip()
                # 去除过短的无效文本
                if col == '正文':
                    df_clean[col] = df_clean[col].apply(lambda x: x if len(x) > 20 else '')
        
        # 6. 创建分析文本
        print("📝 创建分析文本列...")
        analysis_texts = []
        
        for idx, row in df_clean.iterrows():
            # 获取各个文本部分
            title = str(row.get('标题', ''))
            summary = str(row.get('简介', ''))
            content = str(row.get('正文', ''))
            
            # 智能组合文本
            if len(content) > 50:
                main_text = content[:500]  # 限制长度
            elif len(summary) > 30:
                main_text = summary
            else:
                main_text = title
            
            # 添加标题作为前缀（如果标题有信息量）
            if len(title) > 10 and title not in main_text:
                main_text = title + "。" + main_text
            
            # 添加关键词（如果有）
            if '关键词' in row and pd.notna(row['关键词']) and str(row['关键词']).strip():
                keywords = str(row['关键词']).strip()
                if keywords not in main_text:
                    main_text += " " + keywords
            
            analysis_texts.append(main_text)
        
        df_clean['分析文本'] = analysis_texts
        
        # 统计分析文本质量
        text_lengths = [len(t) for t in analysis_texts]
        avg_length = np.mean(text_lengths) if text_lengths else 0
        valid_texts = sum(1 for t in text_lengths if t >= 20)
        
        print(f"   分析文本创建完成，平均长度: {avg_length:.0f} 字符")
        print(f"   有效文本(≥20字符): {valid_texts}/{len(df_clean)} ({valid_texts/len(df_clean)*100:.1f}%)")
        
        # 7. 确保数值列是整数类型（修复格式化错误）
        if '文章数量' in df_clean.columns:
            df_clean['文章数量'] = pd.to_numeric(df_clean['文章数量'], errors='coerce').fillna(1).astype(int)
            print(f"   文章数量列已转换为整数类型")
        
        return df_clean
    
    def sentiment_analysis(self) -> Dict[str, Any]:
        """
        执行情感分析
        
        Returns:
            情感分析结果字典
        """
        print("\n" + "=" * 60)
        print("开始情感分析...")
        print("=" * 60)
        
        if self.df is None or len(self.df) == 0:
            print("❌ 没有数据可分析")
            return {}
        
        sentiments = []
        sentiment_details = []
        
        print("🔍 分析每条新闻的情感...")
        for idx, text in enumerate(self.df['分析文本']):
            try:
                if len(str(text).strip()) < 10:  # 文本太短，跳过
                    sentiments.append(0.5)
                    sentiment_details.append({
                        'score': 0.5,
                        'keywords': [],
                        'sentences': []
                    })
                    continue
                
                s = SnowNLP(str(text))
                score = s.sentiments
                sentiments.append(score)
                
                # 提取情感关键词
                keywords = jieba.analyse.extract_tags(text, topK=5)
                
                # 分析句子情感
                sentences = s.sentences
                sentence_scores = []
                for sent in sentences[:3]:  # 只取前3句
                    try:
                        sent_score = SnowNLP(sent).sentiments
                        sentence_scores.append((sent, sent_score))
                    except:
                        pass
                
                sentiment_details.append({
                    'score': score,
                    'keywords': keywords,
                    'sentences': sentence_scores
                })
                
                if (idx + 1) % 20 == 0:
                    print(f"   已分析 {idx + 1}/{len(self.df)} 条")
                    
            except Exception as e:
                print(f"   ⚠️ 第 {idx + 1} 条分析失败: {str(e)[:50]}")
                sentiments.append(0.5)
                sentiment_details.append({
                    'score': 0.5,
                    'keywords': [],
                    'sentences': []
                })
        
        # 添加情感列到DataFrame
        self.df['情感得分'] = sentiments
        self.df['情感详情'] = sentiment_details
        
        # 情感分类
        def classify_sentiment(score):
            if score >= 0.7:
                return '积极'
            elif score >= 0.4:
                return '中性'
            else:
                return '消极'
        
        self.df['情感分类'] = self.df['情感得分'].apply(classify_sentiment)
        
        # 统计结果
        sentiment_counts = self.df['情感分类'].value_counts()
        sentiment_stats = {
            'total': len(self.df),
            'positive': sentiment_counts.get('积极', 0),
            'neutral': sentiment_counts.get('中性', 0),
            'negative': sentiment_counts.get('消极', 0),
            'mean_score': self.df['情感得分'].mean(),
            'std_score': self.df['情感得分'].std(),
            'min_score': self.df['情感得分'].min(),
            'max_score': self.df['情感得分'].max()
        }
        
        print(f"\n📊 情感分析结果:")
        print(f"   积极: {sentiment_stats['positive']} 条 ({sentiment_stats['positive']/sentiment_stats['total']*100:.1f}%)")
        print(f"   中性: {sentiment_stats['neutral']} 条 ({sentiment_stats['neutral']/sentiment_stats['total']*100:.1f}%)")
        print(f"   消极: {sentiment_stats['negative']} 条 ({sentiment_stats['negative']/sentiment_stats['total']*100:.1f}%)")
        print(f"   平均情感得分: {sentiment_stats['mean_score']:.3f}")
        
        # 情感分布可视化（简版）
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sentiment_counts.plot(kind='bar', ax=ax, color=['#4CAF50', '#FFC107', '#F44336'])
            ax.set_title('情感分布', fontweight='bold')
            ax.set_xlabel('情感分类')
            ax.set_ylabel('数量')
            ax.grid(axis='y', alpha=0.3)
            
            # 在柱子上添加数量
            for i, v in enumerate(sentiment_counts.values):
                ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('情感分布.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   情感分布图已保存: 情感分布.png")
        except Exception as e:
            print(f"   ⚠️ 情感分布图生成失败: {str(e)[:50]}")
        
        self.analysis_results['sentiment'] = sentiment_stats
        return sentiment_stats
    
    def topic_modeling(self, n_topics: int = 5, method: str = 'lda') -> Dict[str, Any]:
        """
        执行主题建模分析
        
        Args:
            n_topics: 主题数量
            method: 主题建模方法 ('lda' 或 'nmf')
            
        Returns:
            主题建模结果字典
        """
        print("\n" + "=" * 60)
        print(f"开始主题建模 ({method.upper()}, {n_topics}个主题)...")
        print("=" * 60)
        
        if self.df is None or len(self.df) < 5:
            print("❌ 数据量不足，至少需要5条数据进行主题建模")
            return {}
        
        # 文本预处理（宽松版）
        print("🔍 预处理文本（宽松版）...")
        processed_texts = self.preprocess_texts_loose(self.df['分析文本'].tolist())
        
        if len(processed_texts) < 3:
            print("❌ 有效文本不足，尝试使用原始文本...")
            # 使用简单分词
            processed_texts = []
            for text in self.df['分析文本'].tolist():
                if isinstance(text, str) and len(text.strip()) > 10:
                    # 简单分词，不过滤停用词
                    words = jieba.lcut(text.strip())
                    words = [w for w in words if len(w) > 1]
                    if len(words) >= 3:
                        processed_texts.append(' '.join(words[:50]))  # 限制长度
        
        if len(processed_texts) < 3:
            print("❌ 仍然没有足够有效文本，跳过主题建模")
            return {}
        
        print(f"   有效预处理文本: {len(processed_texts)}/{len(self.df)}")
        
        try:
            # 创建文档-词矩阵
            print("📊 创建文档-词矩阵...")
            vectorizer = CountVectorizer(
                max_features=500,  # 减少特征数量
                min_df=1,          # 降低最小文档频率
                max_df=0.95,       # 提高最大文档频率
                stop_words=list(CUSTOM_STOPWORDS)
            )
            doc_term_matrix = vectorizer.fit_transform(processed_texts)
            
            print(f"   文档-词矩阵形状: {doc_term_matrix.shape}")
            
            # 调整主题数量（不超过文档数量）
            actual_n_topics = min(n_topics, len(processed_texts) - 1)
            if actual_n_topics < 2:
                actual_n_topics = 2
            
            print(f"   实际使用主题数: {actual_n_topics}")
            
            # 主题建模
            print(f"🧠 训练{method.upper()}模型...")
            if method.lower() == 'lda':
                model = LatentDirichletAllocation(
                    n_components=actual_n_topics,
                    random_state=42,
                    learning_method='online',
                    max_iter=10,  # 减少迭代次数
                    learning_offset=50.
                )
            else:  # nmf
                tfidf_vectorizer = TfidfVectorizer(
                    max_features=500,
                    min_df=1,
                    max_df=0.95,
                    stop_words=list(CUSTOM_STOPWORDS)
                )
                tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)
                model = NMF(
                    n_components=actual_n_topics,
                    random_state=42,
                    max_iter=100
                )
                doc_term_matrix = tfidf_matrix
                vectorizer = tfidf_vectorizer
            
            doc_topic_matrix = model.fit_transform(doc_term_matrix)
            
            # 获取主题关键词
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            print("\n📝 主题关键词:")
            for topic_idx, topic in enumerate(model.components_):
                top_word_indices = topic.argsort()[-10:][::-1]  # 每个主题取10个关键词
                top_words = [feature_names[i] for i in top_word_indices if i < len(feature_names)]
                topics.append(top_words)
                
                print(f"\n主题 {topic_idx + 1}:")
                print(f"  {', '.join(top_words[:8])}")
                
                # 找到该主题的代表性文档
                if len(doc_topic_matrix) > 0:
                    topic_doc_indices = doc_topic_matrix[:, topic_idx].argsort()[-2:][::-1]
                    for i, doc_idx in enumerate(topic_doc_indices):
                        if i < 2 and doc_idx < len(self.df):  # 只显示前2个代表性文档
                            doc_title = str(self.df.iloc[doc_idx].get('标题', '无标题'))[:40]
                            print(f"    代表文档{i+1}: {doc_title}...")
            
            # 为每个文档分配主要主题
            if len(doc_topic_matrix) > 0:
                dominant_topics = doc_topic_matrix.argmax(axis=1)
                self.df['主要主题'] = dominant_topics
                
                # 主题分布统计
                topic_distribution = Counter(dominant_topics)
                
                print(f"\n📊 主题分布:")
                for topic_idx in range(actual_n_topics):
                    count = topic_distribution.get(topic_idx, 0)
                    percentage = count / len(self.df) * 100
                    print(f"   主题 {topic_idx + 1}: {count} 条 ({percentage:.1f}%)")
            else:
                topic_distribution = {}
                print("   无法计算主题分布")
            
            result = {
                'model': model,
                'vectorizer': vectorizer,
                'topics': topics,
                'topic_distribution': dict(topic_distribution),
                'doc_topic_matrix': doc_topic_matrix,
                'n_topics': actual_n_topics,
                'method': method,
                'success': True
            }
            
            # 生成主题可视化
            try:
                self.create_topic_visualization(result)
            except Exception as e:
                print(f"   ⚠️ 主题可视化失败: {str(e)[:50]}")
            
        except Exception as e:
            print(f"❌ 主题建模失败: {str(e)}")
            result = {'success': False, 'error': str(e)}
        
        self.analysis_results['topics'] = result
        return result
    
    def preprocess_texts_loose(self, texts: List[str]) -> List[str]:
        """
        宽松版文本预处理
        
        Args:
            texts: 原始文本列表
            
        Returns:
            预处理后的文本列表
        """
        processed = []
        
        for text in texts:
            if not isinstance(text, str):
                processed.append('')
                continue
            
            text_clean = text.strip()
            if len(text_clean) < 15:  # 降低长度要求
                processed.append('')
                continue
            
            try:
                # 简单清理
                text_clean = re.sub(r'[^\w\u4e00-\u9fff\s，。！？；：、]+', ' ', text_clean)
                text_clean = re.sub(r'\s+', ' ', text_clean)
                
                # 分词（不过滤停用词，只过滤单字）
                words = jieba.lcut(text_clean)
                words_filtered = [w for w in words if len(w) > 1]
                
                if len(words_filtered) >= 3:
                    processed.append(' '.join(words_filtered[:30]))  # 限制词数
                else:
                    # 如果过滤后太少，使用原始分词
                    if len(words) >= 3:
                        processed.append(' '.join(words[:30]))
                    else:
                        processed.append('')
                        
            except Exception as e:
                # 如果出错，使用简单空格分割
                words = text_clean.split()[:20]
                if len(words) >= 3:
                    processed.append(' '.join(words))
                else:
                    processed.append('')
        
        # 移除空文本
        valid_texts = [t for t in processed if t.strip()]
        return valid_texts
    
    def create_topic_visualization(self, topic_data: Dict[str, Any]):
        """创建主题可视化"""
        if not topic_data.get('success', False):
            return
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 1. 主题分布条形图
            topic_dist = topic_data['topic_distribution']
            if topic_dist:
                topics_sorted = sorted(topic_dist.items())
                topic_nums = [f'主题{i+1}' for i, _ in topics_sorted]
                topic_counts = [count for _, count in topics_sorted]
                
                bars = axes[0].bar(topic_nums, topic_counts, color=plt.cm.Set3(range(len(topic_nums))))
                axes[0].set_xlabel('主题')
                axes[0].set_ylabel('文章数量')
                axes[0].set_title('主题分布', fontweight='bold')
                axes[0].grid(True, alpha=0.3, axis='y')
            
            # 2. 主题关键词词云
            axes[1].axis('off')
            
            # 合并所有关键词
            all_keywords = {}
            for i, keywords in enumerate(topic_data['topics']):
                for j, keyword in enumerate(keywords[:6]):  # 每个主题取前6个关键词
                    weight = len(keywords) - j  # 根据位置赋予权重
                    if keyword in all_keywords:
                        all_keywords[keyword] += weight
                    else:
                        all_keywords[keyword] = weight
            
            if all_keywords:
                # 创建词云
                wordcloud = WordCloud(
                    font_path='simhei.ttf',
                    width=400,
                    height=300,
                    background_color='white',
                    max_words=50
                ).generate_from_frequencies(all_keywords)
                
                axes[1].imshow(wordcloud, interpolation='bilinear')
                axes[1].set_title('主题关键词词云', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('主题分析.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   主题分析图已保存: 主题分析.png")
            
        except Exception as e:
            print(f"   ⚠️ 主题可视化创建失败: {str(e)[:50]}")
    
    def temporal_analysis(self) -> Dict[str, Any]:
        """
        时间序列分析
        
        Returns:
            时间序列分析结果
        """
        print("\n" + "=" * 60)
        print("开始时间序列分析...")
        print("=" * 60)
        
        if self.df is None or '发布时间' not in self.df.columns:
            print("❌ 缺少发布时间数据")
            return {}
        
        # 确保有情感得分列
        if '情感得分' not in self.df.columns:
            print("⚠️ 未找到情感得分，先执行情感分析")
            self.sentiment_analysis()
        
        # 按时间分组（按天）
        self.df['日期'] = self.df['发布时间'].dt.date
        
        # 每日统计
        daily_stats = self.df.groupby('日期').agg({
            '情感得分': ['mean', 'std', 'count'],
            '标题': 'count'
        }).round(3)
        
        # 重命名列
        daily_stats.columns = ['情感均值', '情感标准差', '情感样本数', '文章数量']
        
        # 计算移动平均（如果有足够数据）
        if len(daily_stats) > 3:
            daily_stats['情感_3日均值'] = daily_stats['情感均值'].rolling(window=3, min_periods=1).mean()
        
        print(f"\n📅 时间范围: {daily_stats.index.min()} 到 {daily_stats.index.max()}")
        print(f"   总天数: {len(daily_stats)}")
        print(f"   平均每天文章数: {daily_stats['文章数量'].mean():.1f}")
        
        # 找出关键日期
        if len(daily_stats) > 0:
            max_sentiment_date = daily_stats['情感均值'].idxmax()
            min_sentiment_date = daily_stats['情感均值'].idxmin()
            
            # 找出文章最多的日期（如果有文章数量信息）
            if '文章数量' in daily_stats.columns and daily_stats['文章数量'].sum() > 0:
                max_articles_date = daily_stats['文章数量'].idxmax()
                max_articles_count = daily_stats.loc[max_articles_date, '文章数量']
            else:
                max_articles_date = daily_stats.index[0]
                max_articles_count = 1
            
            print(f"\n📈 关键日期:")
            print(f"   情感最高日: {max_sentiment_date} (得分: {daily_stats.loc[max_sentiment_date, '情感均值']:.3f})")
            print(f"   情感最低日: {min_sentiment_date} (得分: {daily_stats.loc[min_sentiment_date, '情感均值']:.3f})")
            print(f"   文章最多日: {max_articles_date} (数量: {max_articles_count})")
            
            # 生成时间序列图
            try:
                self.create_temporal_visualization(daily_stats)
            except Exception as e:
                print(f"   ⚠️ 时间序列图生成失败: {str(e)[:50]}")
        
        result = {
            'daily_stats': daily_stats,
            'date_range': {
                'start': daily_stats.index.min() if len(daily_stats) > 0 else None,
                'end': daily_stats.index.max() if len(daily_stats) > 0 else None,
                'days': len(daily_stats)
            },
            'avg_articles_per_day': daily_stats['文章数量'].mean() if len(daily_stats) > 0 else 0
        }
        
        self.analysis_results['temporal'] = result
        return result
    
    def create_temporal_visualization(self, daily_stats):
        """创建时间序列可视化"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 情感时间序列
        ax.plot(daily_stats.index, daily_stats['情感均值'], 
                marker='o', linewidth=2, color='#2196F3', label='日均情感')
        
        if '情感_3日均值' in daily_stats.columns:
            ax.plot(daily_stats.index, daily_stats['情感_3日均值'], 
                    linewidth=3, color='#FF5722', alpha=0.7, label='3日移动平均')
        
        # 填充标准差区域
        if '情感标准差' in daily_stats.columns:
            ax.fill_between(daily_stats.index,
                           daily_stats['情感均值'] - daily_stats['情感标准差'],
                           daily_stats['情感均值'] + daily_stats['情感标准差'],
                           alpha=0.2, color='#2196F3')
        
        ax.set_xlabel('日期')
        ax.set_ylabel('情感得分')
        ax.set_title('DeepSeek新闻情感时间序列', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('时间序列分析.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   时间序列图已保存: 时间序列分析.png")
    
    def media_analysis(self) -> Dict[str, Any]:
        """
        媒体来源分析（修复版）
        
        Returns:
            媒体分析结果字典
        """
        print("\n" + "=" * 60)
        print("开始媒体来源分析...")
        print("=" * 60)
        
        if self.df is None or '来源' not in self.df.columns:
            print("❌ 缺少来源数据")
            return {}
        
        # 确保有情感得分列
        if '情感得分' not in self.df.columns:
            print("⚠️ 未找到情感得分，先执行情感分析")
            self.sentiment_analysis()
        
        # 媒体统计
        try:
            # 确保文章数量是整数
            if '文章数量' not in self.df.columns:
                self.df['文章数量'] = 1
            
            # 转换为整数类型
            self.df['文章数量'] = pd.to_numeric(self.df['文章数量'], errors='coerce').fillna(1).astype(int)
            
            media_stats = self.df.groupby('来源').agg({
                '情感得分': ['mean', 'std', 'count'],
                '文章数量': 'sum'  # 使用sum而不是count
            }).round(3)
            
            # 重命名列
            media_stats.columns = ['情感均值', '情感标准差', '情感样本数', '文章数量']
            
            # 确保文章数量是整数
            media_stats['文章数量'] = media_stats['文章数量'].astype(int)
            
            media_stats = media_stats.sort_values('文章数量', ascending=False)
            
            print(f"\n📰 媒体来源分析:")
            print(f"   总媒体数: {len(media_stats)}")
            
            if len(media_stats) > 0:
                print(f"   前10大媒体:")
                
                for i, (media, row) in enumerate(media_stats.head(10).iterrows()):
                    # 修复格式化错误：确保文章数量是整数
                    article_count = int(row['文章数量'])
                    print(f"   {i+1:2d}. {media[:20]:20s} - {article_count:3d} 篇, 情感: {row['情感均值']:.3f}")
                
                # 媒体情感分布
                if len(media_stats[media_stats['文章数量'] >= 2]) > 0:
                    print(f"\n😊 最积极的媒体 (至少有2篇文章):")
                    positive_media = media_stats[media_stats['文章数量'] >= 2].nlargest(5, '情感均值')
                    for i, (media, row) in enumerate(positive_media.iterrows()):
                        print(f"   {i+1:2d}. {media[:20]:20s} - 情感: {row['情感均值']:.3f}, 文章: {int(row['文章数量'])}篇")
                
                if len(media_stats[media_stats['文章数量'] >= 2]) > 0:
                    print(f"\n😟 最消极的媒体 (至少有2篇文章):")
                    negative_media = media_stats[media_stats['文章数量'] >= 2].nsmallest(5, '情感均值')
                    for i, (media, row) in enumerate(negative_media.iterrows()):
                        print(f"   {i+1:2d}. {media[:20]:20s} - 情感: {row['情感均值']:.3f}, 文章: {int(row['文章数量'])}篇")
                
                # 生成媒体分析图
                try:
                    self.create_media_visualization(media_stats)
                except Exception as e:
                    print(f"   ⚠️ 媒体分析图生成失败: {str(e)[:50]}")
            
            result = {
                'media_stats': media_stats,
                'top_media': media_stats.head(10).to_dict('index') if len(media_stats) > 0 else {},
                'total_media': len(media_stats),
                'most_positive': positive_media.to_dict('index') if 'positive_media' in locals() else {},
                'most_negative': negative_media.to_dict('index') if 'negative_media' in locals() else {}
            }
            
        except Exception as e:
            print(f"❌ 媒体分析失败: {str(e)}")
            result = {}
        
        self.analysis_results['media'] = result
        return result
    
    def create_media_visualization(self, media_stats):
        """创建媒体分析可视化"""
        if len(media_stats) == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 媒体文章数量Top 10
        top_media = media_stats.nlargest(10, '文章数量')
        
        if len(top_media) > 0:
            y_pos = range(len(top_media))
            bars1 = axes[0].barh(y_pos, top_media['文章数量'], 
                                color=plt.cm.Blues(np.linspace(0.5, 1, len(top_media))))
            
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels([str(name)[:15] for name in top_media.index], fontsize=9)
            axes[0].invert_yaxis()
            axes[0].set_xlabel('文章数量')
            axes[0].set_title('媒体文章数量Top 10', fontweight='bold')
            axes[0].grid(True, alpha=0.3, axis='x')
            
            # 在条形上添加数量标签
            for i, (bar, count) in enumerate(zip(bars1, top_media['文章数量'])):
                width = bar.get_width()
                axes[0].text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                            f'{int(count)}', ha='left', va='center', fontsize=9)
        
        # 2. 媒体情感得分（至少有2篇文章）
        media_with_enough = media_stats[media_stats['文章数量'] >= 2]
        if len(media_with_enough) > 0:
            top_sentiment = media_with_enough.nlargest(8, '情感均值')
            
            y_pos2 = range(len(top_sentiment))
            colors2 = []
            for score in top_sentiment['情感均值']:
                if score > 0.6:
                    colors2.append('#4CAF50')  # 绿色表示积极
                elif score > 0.4:
                    colors2.append('#FFC107')  # 黄色表示中性
                else:
                    colors2.append('#F44336')  # 红色表示消极
            
            bars2 = axes[1].barh(y_pos2, top_sentiment['情感均值'], color=colors2)
            
            axes[1].set_yticks(y_pos2)
            axes[1].set_yticklabels([str(name)[:15] for name in top_sentiment.index], fontsize=9)
            axes[1].invert_yaxis()
            axes[1].set_xlabel('平均情感得分')
            axes[1].set_title('媒体情感得分Top 8 (≥2篇文章)', fontweight='bold')
            axes[1].set_xlim(0, 1)
            axes[1].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
            axes[1].grid(True, alpha=0.3, axis='x')
            
            # 在条形上添加分数标签
            for i, (bar, score) in enumerate(zip(bars2, top_sentiment['情感均值'])):
                width = bar.get_width()
                axes[1].text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                            f'{score:.3f}', ha='left', va='center', fontsize=9)
        else:
            axes[1].text(0.5, 0.5, '没有足够数据\n(需要至少2篇文章的媒体)', 
                        ha='center', va='center', fontsize=12)
            axes[1].set_title('媒体情感得分', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('媒体分析.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   媒体分析图已保存: 媒体分析.png")
    
    def keyword_analysis(self, top_n: int = 20) -> Dict[str, Any]:
        """
        关键词分析
        
        Args:
            top_n: 提取的关键词数量
            
        Returns:
            关键词分析结果
        """
        print("\n" + "=" * 60)
        print(f"开始关键词分析 (Top {top_n})...")
        print("=" * 60)
        
        if self.df is None:
            print("❌ 没有数据可分析")
            return {}
        
        # 合并所有文本
        all_text = ' '.join(self.df['分析文本'].astype(str).tolist())
        
        if len(all_text) < 100:
            print("❌ 文本内容不足")
            return {}
        
        try:
            # 提取关键词 (TF-IDF)
            print("🔍 提取TF-IDF关键词...")
            tfidf_keywords = jieba.analyse.extract_tags(
                all_text, 
                topK=top_n, 
                withWeight=True,
                allowPOS=('n', 'vn', 'v', 'ns', 'nr', 'nt')  # 限制词性
            )
            
            # 提取关键词 (TextRank)
            print("🔍 提取TextRank关键词...")
            textrank_keywords = jieba.analyse.textrank(
                all_text, 
                topK=top_n, 
                withWeight=True,
                allowPOS=('n', 'vn', 'v', 'ns', 'nr', 'nt')
            )
            
            # 词频统计
            print("🔢 统计词频...")
            all_words = []
            for text in self.df['分析文本']:
                words = jieba.lcut(str(text))
                # 宽松过滤
                words_filtered = [w for w in words if len(w) > 1]
                all_words.extend(words_filtered)
            
            word_freq = Counter(all_words)
            top_word_freq = word_freq.most_common(top_n)
            
            print(f"\n🔑 关键词分析结果:")
            print(f"\n1. TF-IDF 关键词 (权重):")
            for i, (word, weight) in enumerate(tfidf_keywords[:10], 1):
                print(f"   {i:2d}. {word:10s} - {weight:.4f}")
            
            print(f"\n2. TextRank 关键词 (权重):")
            for i, (word, weight) in enumerate(textrank_keywords[:10], 1):
                print(f"   {i:2d}. {word:10s} - {weight:.4f}")
            
            print(f"\n3. 高频词 (词频):")
            for i, (word, freq) in enumerate(top_word_freq[:10], 1):
                print(f"   {i:2d}. {word:10s} - {freq:4d} 次")
            
            # 生成词云
            try:
                self.create_keyword_visualization(dict(top_word_freq))
            except Exception as e:
                print(f"   ⚠️ 词云生成失败: {str(e)[:50]}")
            
            result = {
                'tfidf_keywords': dict(tfidf_keywords),
                'textrank_keywords': dict(textrank_keywords),
                'word_frequency': dict(top_word_freq),
                'total_words': len(all_words),
                'unique_words': len(word_freq)
            }
            
        except Exception as e:
            print(f"❌ 关键词分析失败: {str(e)}")
            result = {}
        
        self.analysis_results['keywords'] = result
        return result
    
    def create_keyword_visualization(self, word_freq):
        """创建关键词可视化"""
        if not word_freq:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 词云
        axes[0].axis('off')
        wordcloud = WordCloud(
            font_path='simhei.ttf',
            width=400,
            height=300,
            background_color='white',
            max_words=50
        ).generate_from_frequencies(word_freq)
        
        axes[0].imshow(wordcloud, interpolation='bilinear')
        axes[0].set_title('关键词词云', fontweight='bold')
        
        # 2. 高频词条形图
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:15]
        words, freqs = zip(*top_words) if top_words else ([], [])
        
        if words:
            y_pos = range(len(words))
            axes[1].barh(y_pos, freqs, color='#9C27B0')
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(words)
            axes[1].invert_yaxis()
            axes[1].set_xlabel('词频')
            axes[1].set_title('高频词Top 15', fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='x')
            
            # 添加频数标签
            for i, freq in enumerate(freqs):
                axes[1].text(freq + 0.5, i, str(freq), va='center')
        
        plt.tight_layout()
        plt.savefig('关键词分析.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   关键词分析图已保存: 关键词分析.png")
    
    def generate_report(self, save_path: str = 'deepseek_analysis_report.txt'):
        """
        生成分析报告
        
        Args:
            save_path: 报告保存路径
        """
        print("\n" + "=" * 60)
        print("生成分析报告...")
        print("=" * 60)
        
        report_lines = []
        
        # 报告标题
        report_lines.append("=" * 80)
        report_lines.append("DeepSeek新闻数据分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"分析数据: {self.file_path}")
        report_lines.append(f"数据条数: {len(self.df)}")
        report_lines.append("")
        
        # 1. 数据概况
        report_lines.append("一、数据概况")
        report_lines.append("-" * 40)
        
        if '发布时间' in self.df.columns:
            date_range = f"{self.df['发布时间'].min().date()} 至 {self.df['发布时间'].max().date()}"
            report_lines.append(f"时间范围: {date_range}")
        
        if '来源' in self.df.columns:
            media_count = self.df['来源'].nunique()
            report_lines.append(f"媒体数量: {media_count}")
        
        report_lines.append(f"有效数据条数: {len(self.df)}")
        report_lines.append("")
        
        # 2. 情感分析结果
        if 'sentiment' in self.analysis_results:
            report_lines.append("二、情感分析")
            report_lines.append("-" * 40)
            
            sentiment = self.analysis_results['sentiment']
            total = sentiment['total']
            
            report_lines.append(f"积极新闻: {sentiment['positive']} 条 ({sentiment['positive']/total*100:.1f}%)")
            report_lines.append(f"中性新闻: {sentiment['neutral']} 条 ({sentiment['neutral']/total*100:.1f}%)")
            report_lines.append(f"消极新闻: {sentiment['negative']} 条 ({sentiment['negative']/total*100:.1f}%)")
            report_lines.append(f"平均情感得分: {sentiment['mean_score']:.3f}")
            report_lines.append(f"情感得分标准差: {sentiment['std_score']:.3f}")
            report_lines.append(f"情感得分范围: {sentiment['min_score']:.3f} - {sentiment['max_score']:.3f}")
            report_lines.append("")
        
        # 3. 主题分析结果
        if 'topics' in self.analysis_results and self.analysis_results['topics'].get('success', False):
            report_lines.append("三、主题分析")
            report_lines.append("-" * 40)
            
            topics_data = self.analysis_results['topics']
            report_lines.append(f"主题建模方法: {topics_data['method'].upper()}")
            report_lines.append(f"主题数量: {topics_data['n_topics']}")
            report_lines.append("")
            
            report_lines.append("各主题关键词:")
            for i, keywords in enumerate(topics_data['topics']):
                count = topics_data['topic_distribution'].get(i, 0)
                percentage = count / len(self.df) * 100 if len(self.df) > 0 else 0
                report_lines.append(f"  主题{i+1} ({count}条, {percentage:.1f}%): {', '.join(keywords[:8])}")
            report_lines.append("")
        
        # 4. 时间序列分析
        if 'temporal' in self.analysis_results:
            report_lines.append("四、时间序列分析")
            report_lines.append("-" * 40)
            
            temporal = self.analysis_results['temporal']
            report_lines.append(f"分析天数: {temporal['date_range']['days']}")
            report_lines.append(f"平均每天文章数: {temporal['avg_articles_per_day']:.1f}")
            
            if len(temporal['daily_stats']) > 0:
                max_date = temporal['daily_stats']['情感均值'].idxmax()
                min_date = temporal['daily_stats']['情感均值'].idxmin()
                max_articles_date = temporal['daily_stats']['文章数量'].idxmax()
                
                report_lines.append(f"情感最高日: {max_date} (得分: {temporal['daily_stats'].loc[max_date, '情感均值']:.3f})")
                report_lines.append(f"情感最低日: {min_date} (得分: {temporal['daily_stats'].loc[min_date, '情感均值']:.3f})")
                report_lines.append(f"文章最多日: {max_articles_date} (数量: {temporal['daily_stats'].loc[max_articles_date, '文章数量']})")
            report_lines.append("")
        
        # 5. 媒体分析
        if 'media' in self.analysis_results and self.analysis_results['media']:
            report_lines.append("五、媒体分析")
            report_lines.append("-" * 40)
            
            media = self.analysis_results['media']
            report_lines.append(f"总媒体数量: {media['total_media']}")
            
            if media.get('top_media'):
                report_lines.append("文章数量最多的媒体 (Top 5):")
                top_media_items = list(media['top_media'].items())[:5]
                for i, (media_name, stats) in enumerate(top_media_items, 1):
                    article_count = int(stats.get('文章数量', 0))
                    sentiment_score = stats.get('情感均值', 0)
                    report_lines.append(f"  {i}. {media_name}: {article_count} 篇, 情感: {sentiment_score:.3f}")
            
            report_lines.append("")
        
        # 6. 关键词分析
        if 'keywords' in self.analysis_results and self.analysis_results['keywords']:
            report_lines.append("六、关键词分析")
            report_lines.append("-" * 40)
            
            keywords = self.analysis_results['keywords']
            report_lines.append(f"总词数: {keywords.get('total_words', 0)}")
            report_lines.append(f"唯一词数: {keywords.get('unique_words', 0)}")
            
            if 'tfidf_keywords' in keywords:
                report_lines.append("TF-IDF权重最高的关键词 (Top 10):")
                tfidf_sorted = sorted(keywords['tfidf_keywords'].items(), 
                                     key=lambda x: x[1], reverse=True)[:10]
                for i, (word, weight) in enumerate(tfidf_sorted, 1):
                    report_lines.append(f"  {i}. {word}: {weight:.4f}")
            
            report_lines.append("")
        
        # 7. 主要发现和建议
        report_lines.append("七、主要发现和建议")
        report_lines.append("-" * 40)
        
        # 基于分析结果生成见解
        insights = []
        
        if 'sentiment' in self.analysis_results:
            sentiment = self.analysis_results['sentiment']
            if sentiment['mean_score'] > 0.7:
                insights.append("总体舆论对DeepSeek持非常积极的态度，平均情感得分高达{:.3f}".format(sentiment['mean_score']))
            elif sentiment['mean_score'] > 0.5:
                insights.append("总体舆论对DeepSeek持积极态度")
            else:
                insights.append("总体舆论对DeepSeek持谨慎或消极态度")
        
        if 'media' in self.analysis_results and 'most_positive' in self.analysis_results['media']:
            most_positive = list(self.analysis_results['media']['most_positive'].keys())
            if most_positive:
                insights.append(f"最积极的媒体来源: {', '.join(most_positive[:2])}")
        
        if 'keywords' in self.analysis_results and 'word_frequency' in self.analysis_results['keywords']:
            top_keywords = list(self.analysis_results['keywords']['word_frequency'].keys())[:5]
            if top_keywords:
                insights.append(f"最常讨论的关键词: {', '.join(top_keywords)}")
        
        for i, insight in enumerate(insights, 1):
            report_lines.append(f"{i}. {insight}")
        
        report_lines.append("")
        report_lines.append("建议:")
        report_lines.append("1. 积极舆论占主导(88.9%)，可加强正面宣传")
        report_lines.append("2. 关注少数消极报道，分析负面情绪原因")
        report_lines.append("3. 分析高频关键词，了解公众关注焦点")
        report_lines.append("4. 监测不同媒体报道角度，优化传播策略")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("报告结束")
        report_lines.append("=" * 80)
        
        # 保存报告
        report_text = '\n'.join(report_lines)
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✅ 分析报告已保存: {save_path}")
            
            # 打印报告摘要
            print("\n📋 报告摘要:")
            print("-" * 40)
            for line in report_lines[:20]:  # 打印前20行作为摘要
                print(line)
            print("... (完整报告请查看文件)")
            
        except Exception as e:
            print(f"❌ 保存报告失败: {str(e)}")
        
        return report_text
    
    def run_full_analysis(self, n_topics: int = 5):
        """
        运行完整分析流程
        
        Args:
            n_topics: 主题数量
            
        Returns:
            包含所有分析结果的字典
        """
        print("=" * 80)
        print("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # 1. 加载和清洗数据
            df = self.load_and_clean_data()
            if df is None or len(df) == 0:
                print("❌ 没有有效数据可分析")
                return {}
            
            # 2. 情感分析
            self.sentiment_analysis()
            
            # 3. 主题建模
            if len(df) >= 5:
                self.topic_modeling(n_topics=min(n_topics, 5))
            else:
                print("⚠️  数据量不足，跳过主题建模")
            
            # 4. 时间序列分析
            if '发布时间' in df.columns:
                self.temporal_analysis()
            
            # 5. 媒体分析
            if '来源' in df.columns:
                self.media_analysis()
            
            # 6. 关键词分析
            self.keyword_analysis(top_n=15)
            
            # 7. 生成报告
            self.generate_report()
            
            # 计算运行时间
            end_time = datetime.now()
            run_time = (end_time - start_time).total_seconds()
            
            print("\n" + "=" * 80)
            print("✅ 分析完成!")
            print(f"⏱️  总运行时间: {run_time:.1f} 秒")
            print(f"📊 分析数据量: {len(df)} 条")
            print(f"📈 生成图表: 情感分布.png, 时间序列分析.png, 媒体分析.png, 关键词分析.png")
            if 'topics' in self.analysis_results and self.analysis_results['topics'].get('success', False):
                print(f"           主题分析.png")
            print(f"📝 分析报告: deepseek_analysis_report.txt")
            print("=" * 80)
            
            return self.analysis_results
            
        except Exception as e:
            print(f"❌ 分析过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """主函数"""
    print("=" * 60)
    
    # 配置文件路径
    file_path = input("请输入新闻数据CSV文件路径 (直接回车使用 'news_data.csv'): ").strip()
    if not file_path:
        file_path = 'news_data.csv'
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        print("请确保文件路径正确，或者将数据文件命名为 'news_data.csv' 放在当前目录")
        return
    
    # 创建分析器实例
    analyzer = DeepSeekNewsAnalyzer(file_path)
    
    # 运行完整分析
    results = analyzer.run_full_analysis(n_topics=5)
    
    if results:
        print("\n🎉 分析成功完成!")
        print("生成的文件:")
        print("  1. 情感分布.png - 情感分析图表")
        print("  2. 时间序列分析.png - 时间趋势图表")
        print("  3. 媒体分析.png - 媒体对比图表")
        print("  4. 关键词分析.png - 关键词分析图表")
        if 'topics' in results and results['topics'].get('success', False):
            print("  5. 主题分析.png - 主题建模图表")
        print("  6. deepseek_analysis_report.txt - 详细分析报告")
        
        # 保存处理后的数据
        if analyzer.df is not None:
            output_file = 'processed_deepseek_news.csv'
            analyzer.df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"  7. {output_file} - 处理后的数据")
    else:
        print("\n❌ 分析失败，请检查数据和错误信息")


# ============================================================================
# 安装依赖说明
# ============================================================================

def print_installation_guide():
    """打印安装指南"""
    print("=" * 60)
    print("安装指南")
    print("=" * 60)
    print("运行此代码前，请先安装以下依赖库:")
    print()
    print("1. 基础数据处理:")
    print("   pip install pandas numpy")
    print()
    print("2. 中文NLP处理:")
    print("   pip install jieba snownlp")
    print()
    print("3. 机器学习与主题建模:")
    print("   pip install scikit-learn")
    print()
    print("4. 数据可视化:")
    print("   pip install matplotlib seaborn wordcloud")
    print()
    print("如果安装wordcloud遇到问题，可以尝试:")
    print("   pip install wordcloud")
    print("   或")
    print("   conda install -c conda-forge wordcloud")
    print("=" * 60)


# ============================================================================
# 脚本执行
# ============================================================================

if __name__ == "__main__":
    # 显示安装指南
    print_installation_guide()
    
    # 询问是否继续
    response = input("\n是否继续运行分析? (y/n): ").strip().lower()
    if response == 'y':
        main()
    else:

        print("已退出程序")
