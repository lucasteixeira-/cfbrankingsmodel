"""
LucasTop25 — Índice de Apoio (versão local)
Calcula um índice de currículo/mérito por time FBS, combinando SRS, Elo, SP+ e
FPI (via CollegeFootballData API), pra apoiar o ranking manual semanal.

Uso:
    python lucastop25_indice.py --week 1

Requer um arquivo .env na mesma pasta, com:
    CFBD_API_KEY=sua_chave_aqui
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

# Procura o .env na mesma pasta do script, não importa de onde o comando for rodado
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

API_KEY = os.getenv("CFBD_API_KEY", "")
API_KEY = API_KEY.strip().strip('"').strip("'")
if API_KEY.lower().startswith("bearer "):
    API_KEY = API_KEY[7:].strip()

if not API_KEY:
    sys.exit(
        "CFBD_API_KEY não encontrada ou vazia.\n"
        f"Verifique se existe um arquivo .env em: {SCRIPT_DIR}\n"
        "Conteúdo esperado (sem aspas, sem a palavra Bearer): CFBD_API_KEY=sua_chave_aqui"
    )

print(f"Usando chave: {API_KEY[:4]}...{API_KEY[-4:]} ({len(API_KEY)} caracteres)\n")

BASE_URL = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

SEASON = 2026  # sobrescrito via --season

# Pesos finais (Passo 3 da especificação)
W_WINS = 0.47
W_SOS_NONCONF = 0.08
W_SOS_CONF = 0.25
W_MOV = 0.15
# 5% restante fica pra ajuste manual seu, fora do script

# Combinação da régua de força do adversário (OSP)
W_RESULTADO_REAL = 0.60  # SRS + Elo
W_PREDITIVO = 0.40       # SP+ + FPI

MOV_CAP = 24  # teto de margem de vitória (diminishing returns)

# Nenhum adversário sozinho pode dominar a média de SOS (mesmo espírito do
# teto de MOV) — sem isso, 1 jogo contra um time de elite (ou muito fraco)
# em meio a só 3-4 jogos de não-conferência distorce a média inteira.
SOS_OPP_CAP = 1.5

# "Encolhimento" da média de SOS fora de conferência quando a amostra é
# pequena (normalmente só 3-4 jogos). Equivale a somar K jogos "neutros"
# fantasmas antes de tirar a média — quanto menor N, mais a média é puxada
# pra perto de zero (neutro) até haver jogos suficientes pra confiar nela.
SOS_NONCONF_SHRINKAGE_K = 6

# Desconto aplicado à OSP do adversário nas médias de SOS quando o jogo foi
# DERROTA. Calendário difícil ainda deve contar um pouco mesmo perdendo, mas
# não no mesmo nível de uma vitória — sem isso, um único jogo perdido contra
# um adversário forte (ex: bowl "de compra") infla o SOS quase tanto quanto
# se tivesse sido vencido.
SOS_LOSS_DISCOUNT = 0.25

# Regularização das primeiras semanas: os componentes wins_score, sos_conf e
# mov_capado não têm nenhuma proteção de amostra pequena (diferente do
# sos_nonconf, que já tem shrinkage própria). Da semana 1 até a 4, eles são
# misturados com uma "âncora" de pré-temporada, com peso decrescente —
# zerando na semana 5, quando os componentes voltam a ser 100% currículo real.
PRESEASON_REGULARIZATION_LAST_WEEK = 4


def preseason_shrink_weight(week):
    if week is None or week > PRESEASON_REGULARIZATION_LAST_WEEK:
        return 0.0
    return max(0.0, (PRESEASON_REGULARIZATION_LAST_WEEK + 1 - week) / PRESEASON_REGULARIZATION_LAST_WEEK)


def build_preseason_anchor(data):
    """OSP calculada só com dados de pré-temporada (SRS/Elo do ano anterior +
    SP+/FPI da pré-temporada atual), sem nenhum dado da temporada em curso —
    serve de referência estável pra regularizar as primeiras semanas."""
    srs_prev = safe_series(data["srs_prev_raw"], "rating")
    elo_prev = safe_series(data["elo_prev_raw"], "elo")
    sp_now = safe_series(data["sp_raw"], "rating")
    fpi_now = safe_series(data["fpi_raw"], "fpi")

    anchor = pd.DataFrame(
        {"srs": srs_prev, "elo": elo_prev, "sp_plus": sp_now, "fpi": fpi_now}
    ).reset_index().rename(columns={"index": "team"})

    anchor["z_srs"] = zscore(anchor["srs"])
    anchor["z_elo"] = zscore(anchor["elo"])
    anchor["z_sp"] = zscore(anchor["sp_plus"])
    anchor["z_fpi"] = zscore(anchor["fpi"])
    anchor["result_score"] = anchor[["z_srs", "z_elo"]].mean(axis=1)
    anchor["predictive_score"] = anchor[["z_sp", "z_fpi"]].mean(axis=1)
    anchor["preseason_osp"] = (
        W_RESULTADO_REAL * anchor["result_score"] + W_PREDITIVO * anchor["predictive_score"]
    )
    anchor["z_preseason_osp"] = zscore(anchor["preseason_osp"])

    return anchor.set_index("team")["z_preseason_osp"]

P4_CONFERENCES = {"SEC", "Big Ten", "Big 12", "ACC"}

# Times sem conferência de verdade (ex: Notre Dame). A tag exata vem da API
# em /teams/fbs — validar se bate com "FBS Independents" na primeira rodada
# (dá pra conferir olhando conf_lookup.get("Notre Dame") ou a saída de
# --conference-summary, que já lista essa categoria separadamente).
INDEPENDENT_CONFERENCES = {"FBS Independents"}
P4_INDEPENDENTS = {"Notre Dame"}  # tratado como P4 pra fins de penalidade FCS

TOP10_BONUS = 0.5
TOP25_BONUS = 0.25

# A API não expõe "recebendo votos" diretamente — aproxima usando a própria
# OSP: adversário fora do Top 25 oficial mas com OSP muito alta (perto do
# teto) provavelmente estava na faixa de "recebendo votos" (~26-35), e
# merece algum crédito de qualidade, só que menor que um Top 25 de verdade.
OSP_FRINGE_THRESHOLD = 0.75
RECEIVING_VOTES_BONUS = 0.15
BAD_LOSS_PENALTY = -0.5
BAD_LOSS_RANK_THRESHOLD = 75  # perder pra alguém fora do Top 75 = bad loss

FCS_LOSS_PENALTY_BASE = -1.5
FCS_LOSS_PENALTY_P4_EXTRA = -0.75  # total -2.25 se P4
FCS_WIN_BASE_CREDIT = 0.05
FCS_WIN_CLOSE_MARGIN_THRESHOLD = 14
FCS_WIN_MAX_DISCOUNT = 0.20

# Bônus extra por vitória em jogo de playoff, crescente por fase (identificado
# pelo texto do campo "notes" do jogo). Precisa ser validado na 1ª rodada —
# se o campo "notes" não trouxer esse texto, o bônus simplesmente não é
# aplicado (não quebra o script), e a gente ajusta a forma de detectar a fase.
PLAYOFF_ROUND_BONUS = {
    "first round": 0.3,
    "quarterfinal": 0.5,
    "semifinal": 0.75,
    "national championship": 1.2,
}


# ---------------------------------------------------------------------------
# Cliente da API
# ---------------------------------------------------------------------------

def cfbd_get(endpoint, params=None):
    r = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params or {})
    if not r.ok:
        print(f"Erro {r.status_code} em {endpoint}: {r.text[:300]}")
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Passo 1 — Buscar dados brutos
# ---------------------------------------------------------------------------

def fetch_data(week, season, prev_season, include_postseason=True):
    teams_raw = cfbd_get("/teams/fbs", {"year": season})
    teams_df = pd.DataFrame(teams_raw)[["school", "conference"]].drop_duplicates(subset="school")
    fbs_schools = set(teams_df["school"])

    # Busca temporada regular E pós-temporada (playoff/bowls) separadamente,
    # já que jogos de pós-temporada têm sua própria numeração de semana — e
    # antes disso o script só pegava "regular", perdendo playoff inteiro.
    # Jogos de campeonato de conferência (Big Ten Championship, SEC
    # Championship etc.) são classificados como "regular" pela API — só
    # bowls e CFP entram como "postseason". Por isso, desligar a
    # pós-temporada aqui dá exatamente o corte "campeonatos de conferência
    # decididos, bowls ainda não" sem precisar de nenhum filtro extra.
    games_regular_raw = cfbd_get("/games", {"year": season, "seasonType": "regular"})
    games_post_raw = cfbd_get("/games", {"year": season, "seasonType": "postseason"}) if include_postseason else []

    reg_df = pd.DataFrame(games_regular_raw)
    if not reg_df.empty:
        reg_df["_stage"] = "regular"
    post_df = pd.DataFrame(games_post_raw)
    if not post_df.empty:
        post_df["_stage"] = "postseason"

    games_df = pd.concat([reg_df, post_df], ignore_index=True) if not post_df.empty else reg_df
    if not games_df.empty:
        # Jogos de temporada regular respeitam o corte de semana; jogos de
        # pós-temporada entram inteiros sempre que já tiverem acontecido
        # (não existem antes da hora, então isso não vaza dados do futuro)
        games_df = games_df[(games_df["_stage"] == "postseason") | (games_df["week"] <= week)]

    # SRS e Elo dependem de jogos já disputados — na pré-temporada, esses dois
    # endpoints costumam vir vazios. SP+ e FPI já publicam pré-temporada.
    srs_raw = cfbd_get("/ratings/srs", {"year": season})
    elo_raw = cfbd_get("/ratings/elo", {"year": season})
    sp_raw = cfbd_get("/ratings/sp", {"year": season})
    fpi_raw = cfbd_get("/ratings/fpi", {"year": season})

    # Fallback: ratings finais da temporada anterior
    srs_prev_raw = cfbd_get("/ratings/srs", {"year": prev_season})
    elo_prev_raw = cfbd_get("/ratings/elo", {"year": prev_season})

    try:
        # Sem filtro de semana: traz a AP Poll de TODAS as semanas já publicadas
        # nesta temporada, pra podermos usar o ranking do adversário na época
        # em que o jogo foi disputado (não uma foto única do fim da temporada).
        ap_raw = cfbd_get("/rankings", {"year": season, "seasonType": "regular"})
    except Exception:
        ap_raw = []

    completed = games_df["homePoints"].notna().sum() if not games_df.empty else 0
    print(f"{len(games_df)} jogos carregados até a semana {week} ({completed} já com placar).")
    print(
        f"SRS atual: {len(srs_raw)} times | Elo atual: {len(elo_raw)} times "
        f"(0 é normal na pré-temporada — o script usa fallback da temporada anterior)"
    )

    return {
        "teams_df": teams_df,
        "fbs_schools": fbs_schools,
        "games_df": games_df,
        "srs_raw": srs_raw,
        "elo_raw": elo_raw,
        "sp_raw": sp_raw,
        "fpi_raw": fpi_raw,
        "srs_prev_raw": srs_prev_raw,
        "elo_prev_raw": elo_prev_raw,
        "ap_raw": ap_raw,
    }


# ---------------------------------------------------------------------------
# Passo 2 — OSP (Opponent Strength Proxy), com fallback pra temporada anterior
# ---------------------------------------------------------------------------

def zscore(series):
    std = series.std(ddof=0)
    if not std or pd.isna(std):
        return series * 0
    return (series - series.mean()) / std


def safe_series(raw, value_col):
    if not raw:
        return pd.Series(dtype="float64", name=value_col)
    df = pd.DataFrame(raw)
    if "team" not in df.columns or value_col not in df.columns:
        return pd.Series(dtype="float64", name=value_col)
    # Alguns anos trazem o mesmo time mais de uma vez (ex: realinhamento de
    # conferência no meio da temporada) — agrupar evita índice duplicado.
    return df.groupby("team")[value_col].mean()


def print_computer_model_rankings(data, top_n=25):
    sp_series = safe_series(data["sp_raw"], "rating").sort_values(ascending=False).head(top_n)
    fpi_series = safe_series(data["fpi_raw"], "fpi").sort_values(ascending=False).head(top_n)

    rows = []
    for i in range(top_n):
        sp_team = sp_series.index[i] if i < len(sp_series) else ""
        fpi_team = fpi_series.index[i] if i < len(fpi_series) else ""
        rows.append({"rank": i + 1, "SP+ (puro)": sp_team, "FPI (puro)": fpi_team})

    print("\nModelos matemáticos puros, sem nenhuma mistura com nossa fórmula:")
    print(pd.DataFrame(rows).to_string(index=False))


def build_osp(data):
    srs_now = safe_series(data["srs_raw"], "rating")
    srs_prev = safe_series(data["srs_prev_raw"], "rating")
    srs_final = srs_now.combine_first(srs_prev)

    elo_now = safe_series(data["elo_raw"], "elo")
    elo_prev = safe_series(data["elo_prev_raw"], "elo")
    elo_final = elo_now.combine_first(elo_prev)

    sp_final = safe_series(data["sp_raw"], "rating")
    fpi_final = safe_series(data["fpi_raw"], "fpi")

    osp = pd.DataFrame(
        {"srs": srs_final, "elo": elo_final, "sp_plus": sp_final, "fpi": fpi_final}
    ).reset_index().rename(columns={"index": "team"})

    osp["z_srs"] = zscore(osp["srs"])
    osp["z_elo"] = zscore(osp["elo"])
    osp["z_sp"] = zscore(osp["sp_plus"])
    osp["z_fpi"] = zscore(osp["fpi"])

    osp["result_score"] = osp[["z_srs", "z_elo"]].mean(axis=1)
    osp["predictive_score"] = osp[["z_sp", "z_fpi"]].mean(axis=1)
    osp["osp"] = W_RESULTADO_REAL * osp["result_score"] + W_PREDITIVO * osp["predictive_score"]

    return osp


# ---------------------------------------------------------------------------
# Passo 3 — Referência de Top 10/25 para os bounties
# ---------------------------------------------------------------------------

def build_weekly_ap_ranks(ap_raw):
    """Retorna {semana: {time: rank}} a partir da AP Poll de todas as semanas."""
    weekly = {}
    for entry in ap_raw:
        week = entry.get("week")
        for poll in entry.get("polls", []):
            if poll["poll"] == "AP Top 25":
                weekly[week] = {r["school"]: r["rank"] for r in poll["ranks"]}
    return weekly


def build_bootstrap_ranks(osp_df):
    """Ranking sintético a partir da OSP, usado só quando não existe NENHUMA AP Poll ainda."""
    ranked = osp_df.sort_values("osp", ascending=False).reset_index(drop=True)
    return {row["team"]: i + 1 for i, row in ranked.iterrows()}


class QualityReference:
    """
    Dá o ranking do adversário, com duas variantes que consideram tanto a
    época do jogo quanto como a campanha do adversário terminou:

    - rank_for_bonus (usada em VITÓRIAS): pega o MELHOR ranking entre a época
      do jogo e o final da temporada. Um adversário que só "se revelou" bom
      depois (ex: Tulane subindo no ranking ao longo do ano) ainda concede
      crédito de vitória de qualidade.
    - rank_for_penalty (usada em DERROTAS): pega o PIOR ranking entre os dois
      momentos. Uma derrota que parecia aceitável na época (ex: Alabama x
      Florida State na semana 1) mas envelheceu mal porque o adversário
      desabou depois, passa a ser penalizada como merece.

    Ranking "na época" segue esta prioridade:
    1. AP Poll daquela semana exata, se existir
    2. AP Poll mais recente disponível (cobre pós-temporada e semanas sem poll)
    3. Ranking sintético via OSP (só se não houver NENHUMA AP Poll ainda)
    """

    def __init__(self, weekly_ap_ranks, bootstrap_ranks):
        self.weekly_ap_ranks = weekly_ap_ranks
        self.bootstrap_ranks = bootstrap_ranks
        self.latest_week = max(weekly_ap_ranks) if weekly_ap_ranks else None

    def _rank_at_time(self, opponent, week, is_postseason):
        if not is_postseason and week in self.weekly_ap_ranks:
            ranks = self.weekly_ap_ranks[week]
        elif self.latest_week is not None:
            ranks = self.weekly_ap_ranks[self.latest_week]
        else:
            ranks = self.bootstrap_ranks
        return ranks.get(opponent, 999)

    def _rank_final(self, opponent):
        if self.latest_week is not None:
            return self.weekly_ap_ranks[self.latest_week].get(opponent, 999)
        return self.bootstrap_ranks.get(opponent, 999)

    def rank_for_bonus(self, opponent, week, is_postseason):
        at_time = self._rank_at_time(opponent, week, is_postseason)
        final = self._rank_final(opponent)
        return min(at_time, final)  # menor número = ranking melhor

    def rank_for_penalty(self, opponent, week, is_postseason):
        at_time = self._rank_at_time(opponent, week, is_postseason)
        final = self._rank_final(opponent)
        return max(at_time, final)  # maior número = ranking pior

    @property
    def source_label(self):
        if self.weekly_ap_ranks:
            return f"AP Poll semana-a-semana ({len(self.weekly_ap_ranks)} semanas disponíveis)"
        return "OSP (com fallback pré-temporada)"


# ---------------------------------------------------------------------------
# Passo 4 — Game score por jogo (wins ajustados + bounties + MOV capado)
# ---------------------------------------------------------------------------

def process_games(team, data, osp_lookup, quality_ref, conf_lookup):
    games_df = data["games_df"]
    fbs_schools = data["fbs_schools"]

    team_games = games_df[(games_df["homeTeam"] == team) | (games_df["awayTeam"] == team)]

    game_scores = []
    fcs_adjustment = 0.0
    non_conf_opps = []
    conf_opps = []
    mov_factors = []
    game_log = []

    for _, g in team_games.iterrows():
        if pd.isna(g.get("homePoints")) or pd.isna(g.get("awayPoints")):
            continue  # jogo ainda não aconteceu

        is_home = g["homeTeam"] == team
        opponent = g["awayTeam"] if is_home else g["homeTeam"]
        team_pts = g["homePoints"] if is_home else g["awayPoints"]
        opp_pts = g["awayPoints"] if is_home else g["homePoints"]
        won = team_pts > opp_pts
        margin = team_pts - opp_pts
        neutral = g.get("neutralSite", False)
        game_week = g.get("week")
        is_postseason = g.get("_stage") == "postseason"

        # --- Jogos contra FCS: ajuste lateral, fora dos componentes normais ---
        if opponent not in fbs_schools:
            if won:
                discount = max(0, (FCS_WIN_CLOSE_MARGIN_THRESHOLD - margin) / FCS_WIN_CLOSE_MARGIN_THRESHOLD) * FCS_WIN_MAX_DISCOUNT
                delta = FCS_WIN_BASE_CREDIT - discount
            else:
                delta = FCS_LOSS_PENALTY_BASE
                if conf_lookup.get(team) in P4_CONFERENCES or team in P4_INDEPENDENTS:
                    delta += FCS_LOSS_PENALTY_P4_EXTRA
            fcs_adjustment += delta
            game_log.append({
                "semana": f"pós-{game_week}" if is_postseason else game_week,
                "_ord": (game_week or 0) + (100 if is_postseason else 0),
                "adversário": opponent, "tipo": "FCS",
                "resultado": "V" if won else "D", "placar": f"{int(team_pts)}-{int(opp_pts)}",
                "score_jogo": None, "ajuste_fcs": round(delta, 3),
            })
            continue

        # --- Jogos normais (FBS x FBS) ---
        base = 1 if won else -1

        if neutral:
            local_adj = 1.05
        elif won:
            local_adj = 1.15 if not is_home else 1.0   # vencer fora vale mais
        else:
            local_adj = 1.15 if is_home else 1.0        # perder em casa pesa mais

        opp_osp_raw = osp_lookup.get(opponent, 0.0)
        # Capado só pra uso nas médias de SOS/bônus de qualidade — evita que 1
        # adversário de elite (ou muito fraco) domine sozinho uma amostra pequena
        opp_osp_capped = max(-SOS_OPP_CAP, min(SOS_OPP_CAP, opp_osp_raw))

        mov_factor = min(abs(margin), MOV_CAP) / MOV_CAP
        if won:
            mov_factors.append(mov_factor)  # "margem de VITÓRIA" — derrota não entra aqui

        # Crédito de força do adversário só entra em VITÓRIAS de forma plena
        # (vitória de qualidade). Numa DERROTA, em vez de simplesmente
        # descontar, agora existe um "alívio de qualidade": perder apertado
        # pra um adversário bom pesa bem menos que perder feio (ou perder pra
        # time fraco) — que continua pesando o mesmo de sempre.
        if won:
            score = (base * local_adj) + (opp_osp_capped * 0.3) + (mov_factor * 0.15)
        else:
            closeness = 1 - mov_factor  # jogo apertado -> perto de 1; goleada sofrida -> perto de 0
            quality_relief = max(0, opp_osp_capped) * 0.3 * closeness
            score = (base * local_adj) + quality_relief

        # Vitória: usa o MELHOR ranking entre a época do jogo e o final da
        # temporada (dá crédito por adversário que só se revelou bom depois).
        # Derrota: usa o PIOR ranking entre os dois (pune derrota que parecia
        # aceitável na hora mas envelheceu mal quando o adversário desabou).
        playoff_round_detected = None
        raw_notes = str(g.get("notes") or "")
        if won:
            opp_rank = quality_ref.rank_for_bonus(opponent, game_week, is_postseason)
            if opp_rank <= 10:
                score += TOP10_BONUS
            elif opp_rank <= 25:
                score += TOP25_BONUS
            elif opp_osp_capped >= OSP_FRINGE_THRESHOLD:
                score += RECEIVING_VOTES_BONUS

            if is_postseason:
                notes_lower = raw_notes.lower()
                for keyword, bonus in PLAYOFF_ROUND_BONUS.items():
                    if keyword in notes_lower:
                        score += bonus
                        playoff_round_detected = keyword
                        break
        else:
            opp_rank = quality_ref.rank_for_penalty(opponent, game_week, is_postseason)
            if opp_rank > BAD_LOSS_RANK_THRESHOLD:
                score += BAD_LOSS_PENALTY

        game_scores.append(score)

        # SOS conta a força do adversário mesmo em derrota, mas com desconto
        # que agora escala pela proximidade do jogo (mesma lógica do alívio
        # acima): perder apertado pra um time bom ainda credita algo; levar
        # goleada do mesmo time bom credita quase nada — antes era um desconto
        # fixo (25%) que não diferenciava as duas situações.
        if won:
            sos_contribution = opp_osp_capped
        else:
            closeness = 1 - mov_factor
            sos_contribution = opp_osp_capped * SOS_LOSS_DISCOUNT * closeness

        is_nonconf = conf_lookup.get(team) != conf_lookup.get(opponent)
        if conf_lookup.get(team) in INDEPENDENT_CONFERENCES:
            non_conf_opps.append(sos_contribution)
            conf_opps.append(sos_contribution)
        elif is_nonconf:
            non_conf_opps.append(sos_contribution)
        else:
            conf_opps.append(sos_contribution)

        game_log.append({
            "semana": f"pós-{game_week}" if is_postseason else game_week,
            "_ord": (game_week or 0) + (100 if is_postseason else 0),
            "adversário": opponent,
            "tipo": "não-conf" if is_nonconf else "conf",
            "resultado": "V" if won else "D", "placar": f"{int(team_pts)}-{int(opp_pts)}",
            "opp_osp": round(opp_osp_raw, 3), "mov_factor": round(mov_factor, 3),
            "score_jogo": round(score, 3), "sos_contrib": round(sos_contribution, 3),
            "pós_temp": is_postseason, "fase_detectada": playoff_round_detected,
            "notes_bruto": raw_notes if is_postseason else "",
        })

    # Encolhimento (shrinkage) do SOS de não-conferência: com poucos jogos,
    # puxa a média pra perto de neutro até haver amostra suficiente.
    n_nc = len(non_conf_opps)
    if n_nc > 0:
        raw_mean_nc = np.mean(non_conf_opps)
        sos_nonconf = (n_nc * raw_mean_nc) / (n_nc + SOS_NONCONF_SHRINKAGE_K)
    else:
        sos_nonconf = 0.0

    return {
        "wins_score": np.mean(game_scores) if game_scores else 0.0,
        "sos_nonconf": sos_nonconf,
        "sos_conf": np.mean(conf_opps) if conf_opps else 0.0,
        "mov_capado": np.mean(mov_factors) if mov_factors else 0.0,
        "fcs_adjustment": fcs_adjustment,
        "games_played": len(game_scores),
        "game_log": game_log,
    }


# ---------------------------------------------------------------------------
# Passo 5 — Índice final
# ---------------------------------------------------------------------------

def print_conference_summary(osp_df, conf_lookup):
    df = osp_df.copy()
    df["conference"] = df["team"].map(conf_lookup)
    df = df.dropna(subset=["conference"])
    summary = df.groupby("conference")[["osp", "z_srs", "z_elo", "z_sp", "z_fpi"]].mean()
    summary = summary.sort_values("osp", ascending=False)
    pd.set_option("display.width", 160)
    print("\nForça média por conferência (times FBS, temporada atual):")
    print(summary.round(3).to_string())


def compute_index(data, osp_df, week=None):
    osp_lookup = dict(zip(osp_df["team"], osp_df["osp"]))

    weekly_ap_ranks = build_weekly_ap_ranks(data["ap_raw"])
    bootstrap_ranks = build_bootstrap_ranks(osp_df)
    quality_ref = QualityReference(weekly_ap_ranks, bootstrap_ranks)
    print(f"Referência de qualidade usada: {quality_ref.source_label}")

    conf_lookup = data["teams_df"].set_index("school")["conference"].to_dict()

    results = []
    game_logs = {}
    for team in data["fbs_schools"]:
        r = process_games(team, data, osp_lookup, quality_ref, conf_lookup)
        game_logs[team] = r.pop("game_log")
        r["team"] = team
        results.append(r)

    df = pd.DataFrame(results)
    df = df[df["games_played"] > 0].copy()

    if df.empty:
        print("Nenhum time com jogos concluídos ainda nesta semana.")
        return df, game_logs

    for col in ["wins_score", "sos_nonconf", "sos_conf", "mov_capado"]:
        df[f"z_{col}"] = zscore(df[col])

    shrink = preseason_shrink_weight(week)
    if shrink > 0:
        anchor = build_preseason_anchor(data)
        df["z_preseason_osp"] = df["team"].map(anchor).fillna(0.0)
        print(f"(regularização de início de temporada ativa: {shrink:.0%} âncora de pré-temporada)")
        for col in ["z_wins_score", "z_sos_conf", "z_mov_capado"]:
            df[col] = shrink * df["z_preseason_osp"] + (1 - shrink) * df[col]

    df["contrib_wins"] = W_WINS * df["z_wins_score"]
    df["contrib_sos_nonconf"] = W_SOS_NONCONF * df["z_sos_nonconf"]
    df["contrib_sos_conf"] = W_SOS_CONF * df["z_sos_conf"]
    df["contrib_mov"] = W_MOV * df["z_mov_capado"]
    df["contrib_fcs"] = df["fcs_adjustment"]

    df["indice_final"] = (
        df["contrib_wins"] + df["contrib_sos_nonconf"] + df["contrib_sos_conf"]
        + df["contrib_mov"] + df["contrib_fcs"]
    )

    ranking = df.sort_values("indice_final", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    return ranking, game_logs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_paste_block(df, score_col, conf_lookup, top_n):
    """Bloco rank|time|conferência|pontuação, pensado pra colar direto no
    Monta-Ranking (lucastop25-tier-builder.html) — formato sem ambiguidade,
    já que nome de conferência tem espaço (ex: 'Big Ten')."""
    print(f"\n--- Cole no Monta-Ranking (Top {top_n}) ---")
    for _, row in df.head(top_n).iterrows():
        conf = conf_lookup.get(row["team"], "")
        print(f"{int(row['rank'])}|{row['team']}|{conf}|{row[score_col]:.6f}")
    print("--- fim do bloco ---\n")

def main():
    parser = argparse.ArgumentParser(description="Calcula o índice LucasTop25 pra uma semana.")
    parser.add_argument("--week", type=int, default=1, help="Semana da temporada (default: 1)")
    parser.add_argument("--season", type=int, default=SEASON, help="Ano da temporada (default: 2026)")
    parser.add_argument(
        "--detail", action="store_true",
        help="Mostra o detalhamento por componente (raw + contribuição ponderada) de cada time, "
             "além da pontuação final. Sem essa flag, mostra só rank/time/índice."
    )
    parser.add_argument("--top", type=int, default=25, help="Quantos times mostrar (default: 25)")
    parser.add_argument(
        "--team", type=str, default=None,
        help="Mostra o log jogo-a-jogo de um time específico (nome exato, ex: 'Houston')"
    )
    parser.add_argument(
        "--conference-summary", action="store_true",
        help="Mostra a força média (OSP e cada métrica que a compõe) por conferência"
    )
    parser.add_argument(
        "--no-bowls", action="store_true",
        help="Exclui bowls e CFP do cálculo — mostra o corte 'campeonatos de conferência "
             "decididos, bowls ainda não jogados' (jogos de campeonato de conferência continuam "
             "incluídos, pois são classificados como temporada regular)"
    )
    parser.add_argument(
        "--computer-models", action="store_true",
        help="Mostra o Top 25 de SP+ e FPI puros (sem misturar com nossa fórmula), pra comparar "
             "nosso índice contra modelos matemáticos de terceiros, não só rankings humanos"
    )
    parser.add_argument(
        "--preseason", action="store_true",
        help="Modo pré-temporada: mostra só o ranking por OSP (SRS/Elo do fim da temporada "
             "anterior + SP+/FPI de pré-temporada), sem nenhum componente de currículo (que "
             "exige jogos já disputados). Ignora --week, --detail, --team e --no-bowls."
    )
    args = parser.parse_args()
    prev_season = args.season - 1

    print(f"Calculando índice — temporada {args.season}, semana {args.week}...\n")
    if args.no_bowls:
        print("(pós-temporada excluída — corte pós-campeonatos de conferência, pré-bowls)\n")

    if args.preseason:
        print("(modo pré-temporada — só OSP, sem nenhum componente de currículo)\n")
        data = fetch_data(1, args.season, prev_season, include_postseason=False)
        osp_df = build_osp(data)
        if args.computer_models:
            print_computer_model_rankings(data)
        conf_lookup_preview = data["teams_df"].set_index("school")["conference"].to_dict()
        if args.conference_summary:
            print_conference_summary(osp_df, conf_lookup_preview)
        ranked = osp_df.sort_values("osp", ascending=False).reset_index(drop=True)
        ranked.insert(0, "rank", ranked.index + 1)
        ranked["conference"] = ranked["team"].map(conf_lookup_preview)
        filename = f"LucasTop25_preseason_{args.season}.csv"
        ranked.head(args.top)[["rank", "team", "conference", "osp", "srs", "elo", "sp_plus", "fpi"]].to_csv(filename, index=False)
        print(f"Top {args.top} pré-temporada (por OSP):")
        print(ranked.head(args.top)[["rank", "team", "osp"]].to_string(index=False))
        print_paste_block(ranked, "osp", conf_lookup_preview, args.top)
        print(f"Salvo em: {filename}")
        return

    data = fetch_data(args.week, args.season, prev_season, include_postseason=not args.no_bowls)

    if args.computer_models:
        print_computer_model_rankings(data)
    osp_df = build_osp(data)

    if args.conference_summary:
        conf_lookup_preview = data["teams_df"].set_index("school")["conference"].to_dict()
        print_conference_summary(osp_df, conf_lookup_preview)

    ranking, game_logs = compute_index(data, osp_df, week=args.week)

    if ranking.empty:
        return

    if args.team:
        if args.team not in game_logs:
            print(f"\nTime '{args.team}' não encontrado (confira o nome exato, como aparece na API).")
        else:
            row = ranking[ranking["team"] == args.team]
            if not row.empty:
                print(f"\n{args.team} — rank #{int(row.iloc[0]['rank'])}, índice {row.iloc[0]['indice_final']:.3f}")
            log_df = pd.DataFrame(game_logs[args.team]).sort_values("_ord").drop(columns=["_ord"])
            pd.set_option("display.width", 200)
            pd.set_option("display.max_columns", 20)
            print(log_df.to_string(index=False))
        print()

    all_cols = [
        "rank", "team", "indice_final", "games_played",
        "wins_score", "sos_nonconf", "sos_conf", "mov_capado", "fcs_adjustment",
        "contrib_wins", "contrib_sos_nonconf", "contrib_sos_conf", "contrib_mov", "contrib_fcs",
    ]
    filename = f"LucasTop25_indice_{args.season}_semana{args.week}.csv"
    ranking[all_cols].to_csv(filename, index=False)  # CSV sempre sai completo, com detalhamento

    print(f"\nTop {args.top}:")
    if args.detail:
        detail_cols = [
            "rank", "team", "indice_final", "games_played",
            "wins_score", "sos_nonconf", "sos_conf", "mov_capado", "fcs_adjustment",
            "contrib_wins", "contrib_sos_nonconf", "contrib_sos_conf", "contrib_mov", "contrib_fcs",
        ]
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 20)
        print(ranking[detail_cols].head(args.top).round(3).to_string(index=False))
    else:
        print(ranking[["rank", "team", "indice_final"]].head(args.top).to_string(index=False))

    conf_lookup_final = data["teams_df"].set_index("school")["conference"].to_dict()
    print_paste_block(ranking, "indice_final", conf_lookup_final, args.top)

    print(f"Salvo em: {filename} (sempre com o detalhamento completo, independente da flag --detail)")


if __name__ == "__main__":
    main()
