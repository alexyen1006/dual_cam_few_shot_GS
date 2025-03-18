import os
import pandas as pd
import re

def parse_metrics(file_path):
    """解析指標文件並返回數值"""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            # 使用正則表達式提取數值
            rpe_t = re.search(r'RPE_t:\s*([\d.]+)', content)
            rpe_r = re.search(r'RPE_r:\s*([\d.]+)', content)
            ate = re.search(r'ATE:\s*([\d.]+)', content)
            
            if rpe_t and rpe_r and ate:
                return {
                    'RPE_t': float(rpe_t.group(1)),
                    'RPE_r': float(rpe_r.group(1)),
                    'ATE': float(ate.group(1))
                }
            return None
            
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def compare_metrics(scene_path):
    """比較initialized和optimized的指標"""
    scene_list = os.listdir(scene_path)
    
    # 準備存儲結果的列表
    results = []
    
    for scene in scene_list:
        pose_metric_dir = os.path.join(scene_path, scene, "6_views", "pose", "ours_10000")
        
        # 檢查目錄是否存在
        if not os.path.exists(pose_metric_dir):
            continue
        
        # 讀取文件
        init_file = os.path.join(pose_metric_dir, "pose_eval_initalized_and_GT_wide.txt")
        opt_file = os.path.join(pose_metric_dir, "pose_eval_optimized_and_GT_wide.txt")
        
        init_metrics = parse_metrics(init_file)
        opt_metrics = parse_metrics(opt_file)
        
        if init_metrics and opt_metrics:
            results.append({
                'Scene': scene,
                'Init_RPE_t': init_metrics['RPE_t'],
                'Init_RPE_r': init_metrics['RPE_r'],
                'Init_ATE': init_metrics['ATE'],
                'Opt_RPE_t': opt_metrics['RPE_t'],
                'Opt_RPE_r': opt_metrics['RPE_r'],
                'Opt_ATE': opt_metrics['ATE']
            })
    
    # 創建DataFrame並保存為CSV
    df = pd.DataFrame(results)
    
    # 排序場景名稱
    df['Scene'] = pd.to_numeric(df['Scene'], errors='ignore')
    df = df.sort_values('Scene')
    
    # 設置更好的顯示格式
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    
    # 保存CSV
    df.to_csv('all_pose_wide_comparison_uw_base.csv', index=False)
    
    # 打印表格形式的結果
    print("\nComparison of Initialized vs Optimized Metrics (UW Views):")
    print("=" * 100)
    print(df.to_string(index=False))
    
    # 計算並打印平均值
    means = df.mean(numeric_only=True)
    print("\nAverage Metrics:")
    print("-" * 80)
    print(f"Initialized  - RPE_t: {means['Init_RPE_t']:.4f}, RPE_r: {means['Init_RPE_r']:.4f}, ATE: {means['Init_ATE']:.4f}")
    print(f"Optimized    - RPE_t: {means['Opt_RPE_t']:.4f}, RPE_r: {means['Opt_RPE_r']:.4f}, ATE: {means['Opt_ATE']:.4f}")

if __name__ == "__main__":
    scene_path = "/home/yenalex/dual_cam_few_shot_v0/output/ZoomGS"
    compare_metrics(scene_path)