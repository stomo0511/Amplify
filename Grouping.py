# 創発PBLのグループ分け
# 2025/04/11

# 履修者名簿（CSVファイル）から学部情報 dept と学籍番号 sid を抽出する
import csv

def extract_dept_and_sid(filename):
    dept = []
    sid = []

    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            student_id = row["姓"]  # 「姓」列に学籍番号が記載されている
            if len(student_id) == 8 and student_id[0].isalpha():
                sid.append(student_id)
                dept.append(student_id[0])  # 最初の1文字が学部コード
            else:
                print(f"無効な学籍番号: {student_id}")
                sid.append(None)
                dept.append(None)

    return dept, sid

# 結果をcsvファイルに書き戻す
import pandas as pd

def write_group_to_csv(filename: str, assignment: list[int]) -> None:
    """
    学生ごとのグループ assignment を CSV の「グループ」列に上書き保存

    Parameters:
        filename (str): 元のCSVファイルのパス（このファイルが上書きされます）
        assignment (list[int]): 学生 i の所属グループ番号（長さN）

    Returns:
        None
    """
    # CSVファイルを読み込む
    df = pd.read_csv(filename)

    # assignment の長さと一致しているか確認
    if len(df) != len(assignment):
        raise ValueError("assignment の長さと CSV の行数が一致しません。")

    # 「グループ」列を書き換え
    df["グループ"] = assignment

    # 上書き保存
    df.to_csv(filename, index=False, encoding='cp932')

###############################################################################
# Main
###############################################################################
import sys

if __name__ == "__main__":
    args = sys.argv
    
    if len(args) < 2:
        print("Usage: python3 Grouping.py <受講者名簿.csv>")
        sys.exit(1)
    
    # 履修者名簿の読み込み
    file_name = args[1]
    dept, sid = extract_dept_and_sid(file_name)
    
    N = len(sid) # 学生数
    K = 4 # 1グループの人数
    if (N % K != 0):
        print("人数が4の倍数ではありません。")
        sys.exit(1)
    G = N // K # グループ数（ N % K = 0 を仮定している）
    
    # for i in range(N):
    #     print(f"学籍番号: {sid[i]}, 学部コード: {dept[i]}")
    
    from amplify import VariableGenerator
    gen = VariableGenerator()
    # 変数の定義
    x = gen.array("Binary", shape=(N, G))

    # 制約1
    # 学生は必ず1つのグループに所属する
    from amplify import one_hot

    constraint1 = one_hot(x[:N,:G], axis=1)
    
    # 制約2
    # 各グループの人数はK人
    from amplify import equal_to
    from amplify import sum as amplify_sum

    constraint2 = amplify_sum(
        equal_to(amplify_sum(x[i,k] for i in range(N)), K) for k in range(G)
    )
    
    # コスト関数
    # 一つグループ内でできるだけ異なる学科の学生を集める
    
    cost = 0
    for g in range(G):
        for i in range(N):
            for j in range(i + 1, N):
                if dept[i] == dept[j]:
                    cost += x[i, g] * x[j, g]
    
    # Model
    Consts = constraint1 + constraint2
    model = 10.0*Consts + cost
    
    # Client
    from amplify import FixstarsClient, solve
    client = FixstarsClient()
    client.token = "AE/Pbc9Mo21lDErqmIgJgHhvnM89jvmB37A"

    # print("Solving starts...")
    # Solver
    result = solve(model, client)

    if len(result) == 0:
        print("No solution found.")
        sys.exit(1)
    else:
        print("Solution found.")

    solution = result.best

    values = x.evaluate(result.best.values)
    # print(f"Values of x = {values}")
    
    # グループ分けの結果を配列に格納
    import numpy as np
    
    assignment = np.array([
        next(g for g in range(G) if values[i, g] == 1)
        for i in range(N)
    ])
    # グループ番号は1から始める
    assignment = [g + 1 for g in assignment]

    # CSVファイルに書き戻す
    write_group_to_csv(file_name, assignment)
