# LucasTop25

Índice de apoio (currículo/mérito) pro ranking semanal Top 25 de college football.

## Estrutura

```
scripts/lucastop25_indice.py   -> script principal, roda via Actions ou local
.github/workflows/run-ranking.yml -> aciona o script pela aba Actions do GitHub
docs/                           -> hospedado pelo GitHub Pages
  index.html                    -> página inicial
  tier-builder.html             -> monta o ranking a partir da saída do script
  scoreboard.html                -> visual final, exportável em PNG/PDF/XLSX
data/                            -> resultados salvos (histórico versionado)
```

## Configuração inicial (uma vez só)

1. **Secret da API**: Settings → Secrets and variables → Actions → New repository
   secret → nome `CFBD_API_KEY`, valor: sua chave da collegefootballdata.com.

2. **GitHub Pages**: Settings → Pages → Source: "Deploy from a branch" → Branch:
   `main`, pasta `/docs` → Save. Depois de alguns minutos, as ferramentas ficam
   acessíveis em `https://SEU-USUARIO.github.io/NOME-DO-REPO/`.

3. **Token de escrita** (só quando for usar o botão "Salvar no GitHub" do
   Monta-Ranking): github.com/settings/tokens → Fine-grained tokens → Generate
   new token → Repository access: só este repositório → Permissions: Contents
   = Read and write → Generate. Cole esse token no painel "GitHub" do
   Monta-Ranking (fica salvo só no seu navegador).

## Uso semanal

1. Aba **Actions** do repositório → "Run LucasTop25 Ranking" → Run workflow →
   preenche temporada/semana → Run.
2. Espera terminar (1-2 min), abre `data/<temporada>/<semana>_log.txt` pra ver
   o resultado, ou copia direto do log do próprio Actions.
3. Abre o **Monta-Ranking** (link do Pages), cola a saída, monta o ranking.
4. **Salvar no GitHub** grava sua decisão final em `data/`, ou **Exportar pro
   Scoreboard** baixa o visual já preenchido.
