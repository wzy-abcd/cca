import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# ============================================================
# 1. 通用数据读取函数
# ============================================================
def load_dataset(dataset_path):

    df = pd.read_csv(dataset_path, header=None)
    df.dropna(inplace=True)

    # 根据数据集名称确定标签位置
    dataset_name = dataset_path.split('/')[-1]

    # 定义各数据集的标签位置和特殊处理
    dataset_config = {
        'wine.data': {'label_pos': 0, 'skip_first_col': False},  # 第一列是标签
        'iris.data': {'label_pos': -1, 'skip_first_col': False},  # 最后一列是标签
        'soybean-small.data': {'label_pos': -1, 'skip_first_col': False},
        'waveform.data': {'label_pos': -1, 'skip_first_col': False},
        'waveform+noise.data': {'label_pos': -1, 'skip_first_col': False},
        'zoo.data': {'label_pos': -1, 'skip_first_col': True},  # 跳过第一列（动物名称）
        'bupa.data': {'label_pos': -1, 'skip_first_col': False}
    }

    # 获取配置，默认为最后一列标签，不跳过任何列
    config = dataset_config.get(dataset_name, {'label_pos': -1, 'skip_first_col': False})
    label_pos = config['label_pos']
    skip_first_col = config['skip_first_col']

    print(f"处理数据集: {dataset_name}")
    print(f"  原始数据形状: {df.shape}")
    print(f"  标签位置: {label_pos}, 跳过第一列: {skip_first_col}")

    # 处理特征列
    if skip_first_col:
        # 跳过第一列（通常是ID或名称）
        feature_start_col = 1
    else:
        feature_start_col = 0

    # 提取特征和标签
    if label_pos == -1:  # 最后一列是标签
        X_df = df.iloc[:, feature_start_col:-1]  # 特征列
        y_str = df.iloc[:, -1].values  # 标签列
    else:  # 特定列是标签
        feature_indices = [i for i in range(feature_start_col, df.shape[1]) if i != label_pos]
        X_df = df.iloc[:, feature_indices]  # 特征列
        y_str = df.iloc[:, label_pos].values  # 标签列

    print(f"  特征数据形状: {X_df.shape}")

    # 处理特征数据类型
    X = convert_features_to_numeric(X_df, dataset_name)

    # 把文本标签映射成数字
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    print(f"  最终特征形状: {X.shape}, 标签类别数: {len(le.classes_)}")
    print(f"  类别分布: {dict(zip(le.classes_, np.bincount(y)))}")

    return X, y, le


def convert_features_to_numeric(X_df, dataset_name):  #将特征数据转换为数值类型

    print(f"  正在转换 {dataset_name} 的特征为数值类型...")#f格式化，将变量值嵌入到字符串中

    # 复制数据，避免修改原始数据
    X_numeric = X_df.copy()#没有.copy会直接改变原始数据

    # 遍历每一列进行类型转换
    for col in range(X_numeric.shape[1]):
        try:
            # 尝试直接转换为浮点数
            # errors='raise'表示转换失败时抛出异常
            X_numeric.iloc[:, col] = pd.to_numeric(X_numeric.iloc[:, col], errors='raise')

        except (ValueError, TypeError):
            # 如果直接转换失败（包含字符串），使用标签编码
            try:
                le = LabelEncoder()
                X_numeric.iloc[:, col] = le.fit_transform(X_numeric.iloc[:, col])
                print(f"    列 {col} 使用标签编码转换为数值")

            except Exception as e:
                # 如果标签编码也失败，设为0（最后的手段）
                X_numeric.iloc[:, col] = 0
                print(f"    警告: 列 {col} 无法转换，设为0 (错误: {e})")

    # 返回纯数值特征矩阵
    result = X_numeric.values.astype(float)
    print(f"  转换完成，最终形状: {result.shape}")
    return result


# ============================================================
# 2. 欧式距离函数
# ============================================================
def euclidean_distance(a, b):

    return np.linalg.norm(a - b)


# ============================================================
# 3. 训练：构造性覆盖算法（修改版 - 删除已学习样本）
# ============================================================
def train_cca_version2(X, y):
    # 直接删除已学习的样本

    # 创建数据的副本，因为我们会在训练过程中删除样本
    X_remaining = X.copy()
    y_remaining = y.copy()
    covers = []  # 存储所有覆盖的列表

    print(f"开始训练，初始样本数: {len(X_remaining)}")

    # 循环直到没有剩余样本
    while len(X_remaining) > 0:
        # 1) 从剩余样本中随机选择一个作为覆盖中心
        n_remaining = len(X_remaining)
        k = np.random.choice(n_remaining)  # 随机选择一个索引
        center = X_remaining[k]           # 覆盖中心点
        label_k = y_remaining[k]          # 这个中心点的标签

        # 2) 计算中心点到所有剩余样本的距离
        dists = np.linalg.norm(X_remaining - center, axis=1)

        # 当前样本的同类 / 异类掩码
        mask_hetero = (y_remaining != label_k)
        mask_homo = (y_remaining == label_k)

        # 排除中心点本身（距离为 0）
        mask_homo_others = mask_homo & (dists > 1e-12)

        # ---------- 计算 d1（最近异类距离） ----------
        if np.any(mask_hetero):
            d1_exists = True
            d1 = np.min(dists[mask_hetero])
        else:
            d1_exists = False
            d1 = None  # 只是占位，不参与后面的计算

        # ---------- 计算 d2（同类距离） ----------
        #   - 如果 d1 存在：只考虑 (0, d1) 范围内的同类
        #   - 如果 d1 不存在：考虑所有其他同类样本
        if d1_exists:
            # 在 d1 范围内的同类样本
            mask_homo_in_d1 = mask_homo_others & (dists < d1 - 1e-12)
            if np.any(mask_homo_in_d1):
                d2_exists = True
                d2 = np.max(dists[mask_homo_in_d1])
            else:
                d2_exists = False
                d2 = 0.0
        else:
            # 没有异类时：d2 = 所有其它同类样本中的最大距离
            if np.any(mask_homo_others):
                d2_exists = True
                d2 = np.max(dists[mask_homo_others])
            else:
                d2_exists = False
                d2 = 0.0

        # ---------- “四种情况”计算半径（修改版） ----------
        if d1_exists and d2_exists:
            # 情况 1：d1 存在，d2 存在 -> 正常处理
            radius = (d1 + d2) / 2.0
        elif d1_exists and (not d2_exists):
            # 情况 2：d1 存在，d2 不存在 -> d2 = 0, r = d1 / 2
            radius = d1 / 2.0
        elif (not d1_exists) and d2_exists:
            # 情况 3：d1 不存在，d2 存在 -> 半径 = d2
            radius = d2
        else:
            # 情况 4：d1 不存在，d2 不存在
            # 说明：只剩当前这一点，没有异类也没有其他同类
            # 此时将该覆盖的半径设置为“之前所有覆盖半径的最小值的一半”
            if len(covers) > 0:
                min_prev_radius = min(cov["radius"] for cov in covers)
                radius = min_prev_radius / 2.0
            else:
                # 如果连一个之前的覆盖都没有，只能设为 0（极端情况）
                radius = 0.0

        # 6) 确定当前覆盖包含的样本：同类 + 在半径范围内
        in_cover = (y_remaining == label_k) & (dists <= radius + 1e-12)
        covered_indices = np.where(in_cover)[0]

        current_cover_size = len(covered_indices)

        # 创建覆盖对象并添加到列表
        covers.append({
            "center": center.copy(),      # 覆盖中心
            "radius": radius,            # 覆盖半径
            "label": int(label_k),       # 覆盖标签
            "sample_count": len(covered_indices),
            "size": current_cover_size
        })

        # 7) 删除已被当前覆盖包含的样本
        keep_mask = ~in_cover
        X_remaining = X_remaining[keep_mask]
        y_remaining = y_remaining[keep_mask]

        if len(X_remaining) == 0:
            break

    print(f"训练完成，共生成 {len(covers)} 个覆盖")
    return covers



# ============================================================
# 4. 测试：基于边界距离的分类规则（保持不变）
# ============================================================
def classify_with_boundary_rule(covers, X_test, y_test):
    """
    分类规则：
    1. 如果样本在某个覆盖内部，就用第一个覆盖的标签
    2. 如果不在任何覆盖内部，选择距离边界最近的覆盖的标签
    """
    n = len(X_test)  # 测试样本数量
    y_pred = []  # 存储预测结果
    # id_mask记录哪些样本是可识别的（在覆盖内部）
    id_mask = np.zeros(n, dtype=bool)

    correct_id = 0  # 可识别样本中分类正确的数量
    correct_unid = 0  # 不可识别样本中分类正确的数量

    # 遍历每个测试样本
    for i, x in enumerate(X_test):
        found_inside_cover = False  # 是否找到包含该样本的覆盖
        pred_label = None  # 预测的标签

        # Step 1 & 2：检查样本是否在某个覆盖内部
        # 按照覆盖生成的顺序检查（先找到的覆盖优先级高）
        for cov in covers:
            center = cov["center"]  # 覆盖中心
            r = cov["radius"]  # 覆盖半径
            label = cov["label"]  # 覆盖标签

            # 计算样本到覆盖中心的距离
            d_center = euclidean_distance(x, center)

            # 如果距离小于等于半径，说明在覆盖内部
            if d_center <= r + 1e-12:  # 加个小数值避免浮点数误差
                id_mask[i] = True  # 标记为可识别样本
                pred_label = label  # 使用该覆盖的标签
                found_inside_cover = True
                break  # 找到第一个覆盖就退出

        # Step 3：如果不在任何覆盖内部
        if not found_inside_cover:
            min_boundary_dist = float('inf')  # 初始化最小边界距离为无穷大

            # 遍历所有覆盖，找到边界距离最小的
            for cov in covers:
                center = cov["center"]
                r = cov["radius"]
                label = cov["label"]

                d_center = euclidean_distance(x, center)
                # 计算边界距离：样本到覆盖边界的垂直距离
                boundary_dist = abs(d_center - r)

                # 如果找到更小的边界距离，更新预测标签
                if boundary_dist < min_boundary_dist:
                    min_boundary_dist = boundary_dist
                    pred_label = label

        # 记录预测结果
        y_pred.append(pred_label)

        # 统计正确率
        if pred_label == y_test[i]:  # 如果预测正确
            if found_inside_cover:
                correct_id += 1  # 可识别样本正确数+1
            else:
                correct_unid += 1  # 不可识别样本正确数+1

    y_pred = np.array(y_pred, dtype=int)
    N = len(y_test)  # 总测试样本数
    N_id = int(np.sum(id_mask))  # 可识别样本数
    N_unid = N - N_id  # 不可识别样本数
    total_correct = np.sum(y_pred == y_test)  # 总正确数

    # 返回统计结果
    return {
        "y_pred": y_pred,  # 预测标签
        "N_covers": len(covers),  # 覆盖数量
        "N_id": N_id,  # 可识别样本数
        "N_id_correct": correct_id,  # 可识别样本正确数
        "acc_id": correct_id / N_id if N_id > 0 else 0.0,  # 可识别样本准确率
        "N_unid": N_unid,  # 不可识别样本数
        "N_unid_correct": correct_unid,  # 不可识别样本正确数
        "acc_unid": correct_unid / N_unid if N_unid > 0 else 0.0,  # 不可识别样本准确率
        "total_acc": total_correct / N  # 总准确率
    }


# ============================================================
# 5. K 折交叉验证
# ============================================================
def cross_validate_cca_with_stats(X, y, n_splits=10, random_state=42):
    """
    K折交叉验证：评估模型性能的可靠方法
    """
    # 创建K折交叉验证分割器
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    per_fold_stats = []  # 存储每折的统计结果

    # 进行K折交叉验证
    for fold_id, (train_index, test_index) in enumerate(kf.split(X), start=1):
        # 分割数据
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 数据归一化
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练覆盖模型（使用修改后的版本）
        covers = train_cca_version2(X_train_scaled, y_train)

        # 测试并统计结果
        stats = classify_with_boundary_rule(covers, X_test_scaled, y_test)
        per_fold_stats.append(stats)

        # 打印这折的详细结果
        print(f"Fold {fold_id}:")
        print(f"  覆盖数           = {stats['N_covers']}")
        print(f"  可识别样本数     = {stats['N_id']}")
        print(f"  可识别样本正确数 = {stats['N_id_correct']}  ({stats['acc_id'] * 100:.2f}%)")
        print(f"  不可识别样本数   = {stats['N_unid']}")
        print(f"  不可识别样本正确数 = {stats['N_unid_correct']}  ({stats['acc_unid'] * 100:.2f}%)")
        print(f"  总正确率         = {stats['total_acc'] * 100:.2f}%")
        print("-" * 50)

    # 计算K折的平均结果
    mean_covers = np.mean([s["N_covers"] for s in per_fold_stats])
    mean_id = np.mean([s["N_id"] for s in per_fold_stats])
    mean_id_correct = np.mean([s["N_id_correct"] for s in per_fold_stats])
    mean_acc_id = np.mean([s["acc_id"] for s in per_fold_stats])
    mean_unid = np.mean([s["N_unid"] for s in per_fold_stats])
    mean_unid_correct = np.mean([s["N_unid_correct"] for s in per_fold_stats])
    mean_acc_unid = np.mean([s["acc_unid"] for s in per_fold_stats])
    mean_total_acc = np.mean([s["total_acc"] for s in per_fold_stats])

    # 打印最终的平均结果
    print("=" * 60)
    print(f"{n_splits} 折交叉验证的平均结果：")
    print(f"  平均覆盖数               = {mean_covers:.2f}")
    print(f"  平均可识别样本数         = {mean_id:.2f}")
    print(f"  平均可识别样本正确数     = {mean_id_correct:.2f}")
    print(f"  平均可识别样本正确率(%)  = {mean_acc_id * 100:.2f}")
    print(f"  平均不可识别样本数       = {mean_unid:.2f}")
    print(f"  平均不可识别样本正确数   = {mean_unid_correct:.2f}")
    print(f"  平均不可识别样本正确率(%)= {mean_acc_unid * 100:.2f}")
    print(f"  平均总正确率(%)          = {mean_total_acc * 100:.2f}")
    print("=" * 60)

    return per_fold_stats


# ============================================================
# 6. 主函数：支持多个数据集测试
# ============================================================
if __name__ == "__main__":
    """
    程序的主入口
    """

    # 定义要测试的数据集列表
    datasets = [
        "data/iris.data",
        "data/zoo.data",
        "data/waveform.data",
        "data/waveform-+noise.data",
        "data/bupa.data",
        "data/wine.data",
        "data/soybean-small.data"
    ]

    # 为每个数据集运行测试
    for dataset_path in datasets:
        print("\n" + "=" * 80)
        print(f"开始测试数据集: {dataset_path}")
        print("=" * 80)

        try:
            # 1) 读取数据
            X, y, le = load_dataset(dataset_path)

            # 2) 10 折交叉验证
            stats_per_fold = cross_validate_cca_with_stats(X, y, n_splits=10, random_state=42)

        except Exception as e:
            # 如果某个数据集测试出错，打印错误信息但继续测试其他数据集
            print(f"测试数据集 {dataset_path} 时出错: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n所有数据集测试完成！")