import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import warnings
import subprocess
import random
import yaml
import os
from dataclasses import dataclass
from typing import List, Optional

# PandasのPerformanceWarningを無視する
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

@dataclass
class CaseConfig:
    rain_file: str
    out_dir: str

@dataclass
class GAConfig:
    num_fields: int
    generations: int
    mutation_rate: float
    cases: List[CaseConfig]

def load_config(config_path: str) -> GAConfig:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    cases = [
        CaseConfig(
            rain_file=case['rain_file'],
            out_dir=case['out_dir']
        ) for case in config_data['cases']
    ]

    return GAConfig(
        num_fields=config_data['num_fields'],
        generations=config_data['generations'],
        mutation_rate=config_data['mutation_rate'],
        cases=cases
    )


# 最適化アルゴリズム（GA）参考サイト
# https://qiita.com/pocokhc/items/bca2b374b95c606e110f
# PfGAクラスのプログラムライセンス：https://github.com/pocokhc/metaheuristics/tree/main
class PfGA():
    def __init__(self, mutation=0.1):
        self.mutation = mutation  # 突然変異の確率

    def init(self, problem):
        self.problem = problem
        self.individuals = []  # 局所集団(local individuals)
        for _ in range(2):
            self.individuals.append(problem.create())

    def step(self):
        print(f"Starting step with {len(self.individuals)} individuals.")
        if len(self.individuals) < 2:
            self.individuals.append(self.problem.create())

        p1 = self.individuals.pop(random.randint(0, len(self.individuals)-1))
        p2 = self.individuals.pop(random.randint(0, len(self.individuals)-1))

        c1, c2 = self._cross(p1, p2)
        
        if p1.getScore() < p2.getScore():
            p_min = p1
            p_max = p2
        else:
            p_min = p2
            p_max = p1
        if c1.getScore() < c2.getScore():
            c_min = c1
            c_max = c2
        else:
            c_min = c2
            c_max = c1

        if c_min.getScore() >= p_max.getScore():
            # 子2個体がともに親の2個体より良かった場合
            # 子2個体及び適応度の良かった方の親個体計3個体が局所集団に戻り、局所集団数は1増加する。
            self.individuals.append(c1)
            self.individuals.append(c2)
            self.individuals.append(p_max)
        elif p_min.getScore() >= c_max.getScore():
            # 子2個体がともに親の2個体より悪かった場合
            # 親2個体のうち良かった方のみが局所集団に戻り、局所集団数は1減少する。
            self.individuals.append(p_max)
        elif p_max.getScore() >= c_max.getScore() and p_min.getScore() <= c_max.getScore():
            # 親2個体のうちどちらか一方のみが子2個体より良かった場合
            # 親2個体のうち良かった方と子2個体のうち良かった方が局所集団に戻り、局所集団数は変化しない。
            self.individuals.append(c_max)
            self.individuals.append(p_max)
        elif c_max.getScore() >= p_max.getScore() and c_min.getScore() <= p_max.getScore():
            # 子2個体のうちどちらか一方のみが親2個体より良かった場合
            # 子2個体のうち良かった方のみが局所集団に戻り、全探索空間からランダムに1個体選んで局所集団に追加する。局所集団数は変化しない。
            self.individuals.append(c_max)
            self.individuals.append(self.problem.create())
        else:
            raise ValueError("not comming")
        print(f"Step completed. {len(self.individuals)} individuals in population.")

    def _cross(self, parent1, parent2):
        genes1 = parent1.getArray()  # 親1の遺伝子情報
        genes2 = parent2.getArray()  # 親2の遺伝子情報
        
        # 子の遺伝子情報
        c_genes1 = []
        c_genes2 = []
        for i in range(len(genes1)):  # 各遺伝子を走査
            # 50%の確率で遺伝子を入れ替える
            if random.random() < 0.5:
                c_gene1 = genes1[i]
                c_gene2 = genes2[i]
            else:
                c_gene1 = genes2[i]
                c_gene2 = genes1[i]

            # 突然変異
            if random.random() < self.mutation:
                c_gene1 = self.problem.randomVal()
            if random.random() < self.mutation:
                c_gene2 = self.problem.randomVal()

            c_genes1.append(c_gene1)
            c_genes2.append(c_gene2)

        # 遺伝子をもとに子を生成
        childe1 = self.problem.create(c_genes1)
        childe2 = self.problem.create(c_genes2)
        return childe1, childe2

# 問題定義用のクラス
class WaterLevelProblem:
    def __init__(self, num_fields, out_dir):
        self.num_fields = num_fields  # 水田の数
        self.out_dir = out_dir

    def create(self, genes=None):
        if genes is None:
            genes = [random.randint(0, 1) for _ in range(self.num_fields)]
        return Individual(genes, problem=self)

    def randomVal(self):
        return random.randint(0, 1)

    def evaluate(self, individual):
        with open('ga.txt', 'w') as file:
            file.write(' '.join(map(str, individual.getArray())))
        # 実行ファイルを呼び出し、結果を得る
        result = subprocess.run(['./unst.exe'], text=True)
        return self.read_hmax_value()
    
    def read_hmax_value(self):
        file_path = f'{self.out_dir}/hmax.dat'
        hmax_values = []
        with open(file_path, 'r') as file:
            for line in file:
                # 'time' を含む行をスキップ
                if 'time' in line:
                    continue
                # 行から数値のリストを取得
                try:
                    values = line.split()
                    # 数値として変換可能なものだけをリストに追加
                    hmax_values.extend([float(value) for value in values if value.replace('.', '', 1).isdigit()])
                except ValueError:
                    # 数値変換に失敗した場合はこの行をスキップ
                    continue

        hmax = np.array(hmax_values)
        indices = [31582,31583,31586,31601,31602,31670,31672,31692,31694,31696,31697,31698,31699,31700,31701,31702,31703,31705,31706,31721,31727,31728,31731,31735,31736,31742] #結合部
        indices = [x - 1 for x in indices]  # pythonのindexは0から始まるため
        print(indices)
        
        evaluate_value = np.average(hmax[indices])
        print(f'evaluate_value {evaluate_value}')
        return -1 * evaluate_value  # 負の値として返す

# 個体クラス
class Individual:
    def __init__(self, genes, problem=None):
        self.genes = genes
        self.score = None
        self.problem = problem

    def getScore(self):
        if self.score is None:
            if self.problem is None:
                raise ValueError("Problem instance not set for this individual")
            self.score = self.problem.evaluate(self)
        return self.score

    def getArray(self):
        return self.genes

# 可視化機能を追加したPfGAクラス
class PfGAVisualized(PfGA):
    def __init__(self, mutation=0.1, generations=30):
        super().__init__(mutation)
        self.scores_history = []
        self.best_individual = None
        self.best_score = -float('inf')
        self.generations = generations
        self.all_individuals_history = []

    def step(self, generation):
        super().step()
        current_best_individual = max(self.individuals, key=lambda ind: ind.getScore())
        current_best_score = current_best_individual.getScore()

        self.scores_history.append(current_best_score)
        self.all_individuals_history.append([ind.getScore() for ind in self.individuals])

        if current_best_score > self.best_score:
            self.best_individual = current_best_individual
            self.best_score = current_best_score
        print(f"Generation {generation+1}/{self.generations} - Best score: {self.best_score}")
        print(f"Best individual's genes: {self.best_individual.getArray()}")

    def plot_scores(self, case_num):
        # 各世代の最高スコアのみをプロット
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot([-score for score in self.scores_history], marker='o', linestyle='-', label='Best Score')
        ax.set_title('Best Score Improvement Over Time')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'out/case{case_num}/best_score_improvement.png')
        plt.close(fig)

        # 各世代の全ての個体のスコアをプロット
        fig, ax = plt.subplots(figsize=(10, 6))
        markers = ['x', 'o', '^', 's', 'd', 'v', '<', '>', 'p', 'h']
        colors = ['gray', 'blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'olive', 'cyan']
        for generation, individuals in enumerate(self.all_individuals_history):
            ax.scatter(
                [generation] * len(individuals),
                [-score for score in individuals],
                marker=markers[generation % len(markers)],
                color=colors[generation % len(colors)],
                alpha=0.5,
                label=f'Generation {generation+1}'
            )
        ax.set_title('Score Distribution Over Time')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'out/case{case_num}/score_distribution.png')
        plt.close(fig)

def update_cntl_file(rain_file, out_dir):
    with open('cntl.dat', 'r') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        if line.startswith('data/rain/'):
            updated_lines.append(f'{rain_file}\n')
        elif 'out/' in line and ('h.dat' in line or 'hmax.dat' in line):
            if 'h.dat' in line:
                updated_lines.append(f'{out_dir}/h.dat\n')
            elif 'hmax.dat' in line:
                updated_lines.append(f'{out_dir}/hmax.dat\n')
        else:
            updated_lines.append(line)

    with open('cntl.dat', 'w') as file:
        file.writelines(updated_lines)

def ensure_output_directories(cases: List[CaseConfig]):
    """Ensure all output directories exist"""
    for case in cases:
        os.makedirs(case.out_dir, exist_ok=True)

def run_genetic_algorithm(config: GAConfig):
    """Run genetic algorithm for all specified cases"""
    ensure_output_directories(config.cases)

    for case in config.cases:
        print(f"\nProcessing case with rain file: {case.rain_file}")
        update_cntl_file(case.rain_file, case.out_dir)
        
        problem = WaterLevelProblem(config.num_fields, case.out_dir)
        ga = PfGAVisualized(
            mutation=config.mutation_rate,
            generations=config.generations
        )
        ga.init(problem)

        for generation in range(config.generations):
            ga.step(generation)
        
        # 最終的な最良個体とスコアを出力
        final_best_individual = max(ga.individuals, key=lambda ind: ind.getScore())
        print(f"Case {case.out_dir} - Final Best Individual: {final_best_individual.getArray()}")
        print(f"Case {case.out_dir} - Final Score: {final_best_individual.getScore()}")

        # opt_paddy.csvを保存
        with open(f"{case.out_dir}/opt_paddy.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(final_best_individual.getArray())

        # スコアの履歴をプロット
        case_num = os.path.basename(case.out_dir).replace('case', '')
        ga.plot_scores(case_num)

if __name__ == "__main__":
    config = load_config('config.yml')
    run_genetic_algorithm(config)