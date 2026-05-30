# 导入所需的Python库
# numpy: 用于科学计算，处理数组和矩阵
# pandas: 用于数据处理和分析
# sklearn: 机器学习库，提供各种机器学习算法和工具
import numpy as np
import pandas as pd#(导入一个完整模块并起一个别名)
from sklearn.model_selection import KFold#（导入模块的某个功能）
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# ============================================================
# 1. 通用数据读取函数
# ============================================================
def load_dataset(dataset_path):#输入参数为数据文件路径

    # 用pandas读取CSV文件，header=None表示文件没有表头（列名）
    df = pd.read_csv(dataset_path, header=None)
    # 删除空行（包含缺失值的行）
    df.dropna(inplace=True)

    # 从文件路径中提取数据集名称，比如从"data/iris.data"中提取"iris.data"
    dataset_name = dataset_path.split('/')[-1]

    # 定义每个数据集的配置信息：标签位置和是否跳过第一列（wine的标签位于第一列，zoo的第一列是动物名称，直接跳过）
    dataset_config = {
        'wine.data': {'label_pos': 0, 'skip_first_col': False},  # 第一列是标签
        'iris.data': {'label_pos': -1, 'skip_first_col': False},  # 最后一列是标签
        'soybean-small.data': {'label_pos': -1, 'skip_first_col': False},
        'waveform.data': {'label_pos': -1, 'skip_first_col': False},
        'waveform+noise.data': {'label_pos': -1, 'skip_first_col': False},
        'zoo.data': {'label_pos': -1, 'skip_first_col': True},  # 跳过第一列（动物名称）
        'bupa.data': {'label_pos': -1, 'skip_first_col': False}
    }#数据集配置字典

    # 获取当前数据集的配置，如果找不到就用默认配置（.get获取字典中的值，没有用默认值）
    config = dataset_config.get(dataset_name, {'label_pos': -1, 'skip_first_col': False})
    label_pos = config['label_pos']  # 获取标签所在列的位置
    skip_first_col = config['skip_first_col']  #获取 是否跳过第一列

    # 打印数据集信息
    print(f"处理数据集: {dataset_name}")
    print(f"  原始数据形状: {df.shape}") #（个数，维数）
    print(f"  标签位置: {label_pos}, 跳过第一列: {skip_first_col}")

    # 确定特征数据从哪一列开始
    if skip_first_col:
        # 跳过第一列
        feature_start_col = 1
    else:
        # 从第0列开始（正常情况）
        feature_start_col = 0

    # 提取特征数据(X)矩阵和标签数据(y)数组
    if label_pos == -1:  # 如果标签在最后一列
        # （python切片规则：左闭右开）
        X_df = df.iloc[:, feature_start_col:-1]  # 特征：所有行，从开始列到倒数第二列
        y_str = df.iloc[:, -1].values  # 标签：所有行，最后一列
    else:  # 如果标签在特定列（比如wine数据集在第一列）
        # 创建特征列的索引列表，排除标签列（从起始列到结束列遍历找到所有非标签列）
        feature_indices = [i for i in range(feature_start_col, df.shape[1]) if i != label_pos]
        X_df = df.iloc[:, feature_indices]  # 特征数据（所有行+特征列）
        y_str = df.iloc[:, label_pos].values  # 标签数据（所有行+标签列）

    print(f"  特征数据形状: {X_df.shape}")

    # 将特征数据转换为数值类型（因为机器学习算法只能处理数字）
    X = convert_features_to_numeric(X_df, dataset_name)


    le = LabelEncoder()#创建标签编码器le是一个LabelEncoder对象，记住标签的映射规则
    y = le.fit_transform(y_str)#学习标签的映射规则并应用规则进行转换

    # 特征的类别及类别数
    print(f"  最终特征形状: {X.shape}, 标签类别数: {len(le.classes_)}")
    # 统计每个标签和其出现的次数
    #zip将两个列表按位置进行配对，dict将列表转换为字典，输出（类别名：数量）
    print(f"  类别分布: {dict(zip(le.classes_, np.bincount(y)))}")

    """

    返回:
        X: 特征数据（数字矩阵）
        y: 标签数据（数字数组）
        le: 标签编码器，用于将文字标签转成数字
    """
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
def euclidean_distance(a, b):#参数: a, b: 两个点的坐标（数组）
    #返回:两点之间的距离
    return np.linalg.norm(a - b)


# ============================================================
# 3. 训练：构造性覆盖算法（不升维 + 欧式距离）
# ============================================================
def train_cca_version1(X, y):
    # 构造性覆盖算法（Constructive Covering Algorithm）

    n_samples = X.shape[0]  # 布尔数组：样本数量（获取行数）

    # learned数组记录每个样本是否已经被某个覆盖学习过
    # False表示还没被学习，True表示已经被学习
    learned = np.zeros(n_samples, dtype=bool)
    covers = []  # 存储所有覆盖的列表

    # 循环直到所有样本都被学习过
    while not np.all(learned):
        # 1) 从尚未学习的样本中随机选择一个作为覆盖中心
        candidates = np.where(~learned)[0]  # 找到所有未学习样本的索引，返回索引
        k = np.random.choice(candidates)  #从未被学习过的样本中随机选择一个
        center = X[k]  # 覆盖中心点
        label_k = y[k]  # 这个中心点的标签

        # 2) 计算中心点到所有样本的距离
        dists = np.linalg.norm(X - center, axis=1)#广播机制：center被复制成与X相同的形状（范数求解）

        # 3) 计算d1：到最近的"异类或已学习样本"的距离
        # 异类：标签不同的样本
        # 已学习：已经被其他覆盖包含的样本
        mask_d1 = (y != label_k) | learned

        d1 = np.min(dists[mask_d1])  # 找到最小距离

        # 4) 计算d2：在d1范围内，到最远的"同类且未学习样本"的距离
        mask_d2 = (y == label_k) & (dists < d1 - 1e-12)
        if np.any(mask_d2):
            d2 = np.max(dists[mask_d2])  # 找到最大距离
        else:
            # 如果在d1范围内没有其他同类未学习样本
            d2 = 0.0  # 只有中心点自己

        # 5) 计算覆盖半径：取d1和d2的平均值（折中半径法）
        radius = (d1 + d2) / 2.0

        # 6) 确定当前覆盖包含的样本
        # 条件：未学习 + 同类 + 在半径范围内
        in_cover = (~learned) & (y == label_k) & (dists <= radius + 1e-12)
        covered_indices = np.where(in_cover)[0]  # 找到满足条件的样本索引

        # 创建覆盖对象并添加到列表
        covers.append({
            "center": center,  # 覆盖中心
            "radius": radius,  # 覆盖半径
            "label": int(label_k),  # 覆盖标签
            "sample_count": len(covered_indices),  # 覆盖内样本个数
            "indices": covered_indices  # 覆盖包含的样本索引
        })

        # 标记这些样本为"已学习"
        learned[covered_indices] = True

    #返回:covers: 覆盖列表，每个覆盖包含中心点、半径和标签
    return covers

# ============================================================
# 4. 测试：基于边界距离的分类规则（精简版：只统计总正确率）
# ============================================================
# ============================================================
# 4.1 预测函数：边界规则（不需要y_test，用于集成投票）
# ============================================================
def predict_with_boundary_rule(covers, X_test):
    """
    对 X_test 做预测：
    1) 若落在某个覆盖内部 -> 用第一个覆盖的标签
    2) 否则 -> 用边界距离 abs(d_center - r) 最小的覆盖标签
    """
    y_pred = []

    for x in X_test:
        pred_label = None

        # 先看是否落在某个覆盖内部（先到先得）
        for cov in covers:
            center = cov["center"]
            r = cov["radius"]
            label = cov["label"]

            d_center = euclidean_distance(x, center)
            if d_center <= r + 1e-12:
                pred_label = label
                break

        # 不在任何覆盖内：边界最近
        if pred_label is None:
            min_boundary_dist = float("inf")
            for cov in covers:
                center = cov["center"]
                r = cov["radius"]
                label = cov["label"]

                d_center = euclidean_distance(x, center)
                boundary_dist = abs(d_center - r)

                if boundary_dist < min_boundary_dist:
                    min_boundary_dist = boundary_dist
                    pred_label = label

        y_pred.append(pred_label)

    return np.array(y_pred, dtype=int)


# ============================================================
# 4.2 集成分类：多组covers + 多数投票（v-CCA核心）
# ============================================================
def classify_ensemble_with_majority(covers_list, X_test, y_test):
    """
    covers_list: [covers_1, covers_2, ... covers_M]
    每组covers预测一次，然后多数投票
    只返回：平均覆盖数 + 总正确率
    """
    M = len(covers_list)
    N = len(X_test)

    all_preds = []
    for covers in covers_list:
        all_preds.append(predict_with_boundary_rule(covers, X_test))

    all_preds = np.vstack(all_preds)  # (M, N)

    ensemble_pred = np.zeros(N, dtype=int)
    for i in range(N):
        votes = all_preds[:, i]
        counts = np.bincount(votes)
        ensemble_pred[i] = np.argmax(counts)

    total_acc = np.mean(ensemble_pred == y_test)
    mean_covers = np.mean([len(c) for c in covers_list])

    return {
        "N_covers": mean_covers,   # 平均覆盖数
        "total_acc": total_acc
    }


# ============================================================
# 5. K折交叉验证（v-CCA：每折生成n_generate组covers投票）
# ============================================================
def cross_validate_vcca(X, y, n_splits=10, n_generate=20, random_state=42):
    """
    v-CCA:
      - 每折训练集生成 n_generate 组 covers（基于 train_cca_version1）
      - 对测试集做多数投票
      - 只输出：平均覆盖数 + 总正确率
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    per_fold_stats = []

    # 用固定RandomState来生成每次训练用的seed，保证“可复现+多样性”
    rng = np.random.RandomState(random_state)

    for fold_id, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 归一化
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # === 每折生成 n_generate 组 covers ===
        covers_list = []
        for g in range(n_generate):
            # 每次生成前设置不同seed，确保cover生成不同
            np.random.seed(rng.randint(0, 2**31 - 1))
            covers_g = train_cca_version1(X_train_scaled, y_train)
            covers_list.append(covers_g)

        # === 投票集成 ===
        stats = classify_ensemble_with_majority(covers_list, X_test_scaled, y_test)
        per_fold_stats.append(stats)

        print(f"Fold {fold_id}:")
        print(f"  覆盖数(平均) = {stats['N_covers']:.2f}")
        print(f"  总正确率     = {stats['total_acc'] * 100:.2f}%")
        print("-" * 50)

    mean_covers = np.mean([s["N_covers"] for s in per_fold_stats])
    mean_total_acc = np.mean([s["total_acc"] for s in per_fold_stats])

    print("=" * 60)
    print(f"{n_splits} 折 v-CCA 平均结果（每折 {n_generate} 组 covers 投票）：")
    print(f"  平均覆盖数 = {mean_covers:.2f}")
    print(f"  平均正确率 = {mean_total_acc * 100:.2f}%")
    print("=" * 60)

    return per_fold_stats

# ============================================================
# 6. 主函数：支持多个数据集测试
# ============================================================
if __name__ == "__main__":
    """
    程序的主入口
    当直接运行这个Python文件时，会执行这里的代码
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
            stats_per_fold = cross_validate_vcca(X, y, n_splits=10, n_generate=30, random_state=42)


        except Exception as e:
            # 如果某个数据集测试出错，打印错误信息但继续测试其他数据集
            print(f"测试数据集 {dataset_path} 时出错: {e}")
            import traceback

            traceback.print_exc()  # 打印详细的错误信息
            continue  # 继续下一个数据集

    print("\n所有数据集测试完成！")