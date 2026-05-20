# palazzo — guide rapide (FR)

`palazzo` est un serveur MCP qui donne à vos agents (Claude Code, GitHub
Copilot, etc.) une **mémoire sémantique à long terme** partagée, indexée dans
Qdrant. Une instance tourne en continu sur la VM dev à `http://10.17.0.142:8089/mcp`.

Le principe : au lieu de ré-expliquer le contexte à chaque session, vous le
**stockez** une fois, et les sessions suivantes le **retrouvent** par recherche
sémantique.

---

## 1. Installation

### Claude Code (CLI)

Une seule commande dans votre terminal :

```bash
claude mcp add --transport http palazzo http://10.17.0.142:8089/mcp
```

Vérifier que le serveur est bien connecté :

```bash
claude mcp list
```

Vous devriez voir :

```
palazzo: http://10.17.0.142:8089/mcp (HTTP) - ✓ Connected
```

> ⚠️ Le serveur est accessible uniquement depuis le réseau Tailscale de
> l'entreprise. Si vous voyez `✗ Failed to connect`, vérifiez que Tailscale
> est actif sur votre machine et que vous êtes authentifié sur le tailnet.

### GitHub Copilot (VS Code)

```bash
code --add-mcp '{"name":"palazzo","type":"http","url":"http://10.17.0.142:8089/mcp"}'
```

Ou éditer `.vscode/mcp.json` dans votre workspace :

```json
{
  "servers": {
    "palazzo": {
      "type": "http",
      "url": "http://10.17.0.142:8089/mcp"
    }
  }
}
```

### Vérification

Dans une nouvelle session Claude Code, demandez simplement :

> « Utilise `palace_status` pour me montrer l'état de la mémoire. »

Si l'agent renvoie un total de points + une ventilation par *wing* / *hall* /
*category*, c'est bon.

---

## 2. Comment stocker (write)

Les mémoires sont structurées selon quatre axes — pas besoin de les retenir par
cœur, l'agent connaît le schéma et vous guide :

| Champ | Valeurs | Exemple |
|---|---|---|
| `category` | texte libre — conventionnellement `person` · `career` · `technical` · `infrastructure` · `project-memory` · `vibe` · `project` | `technical` |
| `wing` | texte libre — conventionnellement `projects` · `infrastructure` · `personal` · `career` · `vibe` | `projects` |
| `room` | texte libre — nom du projet ou du sujet | `mon-projet`, `api-paiement`, `équipe-backend` |
| `hall` | texte libre — conventionnellement `facts` · `events` · `decisions` · `discoveries` · `preferences` | `decisions` |

Les quatre axes sont du texte libre — les valeurs ci-dessus sont des conventions, pas des contraintes. Organisez votre palais comme bon vous semble.

### Stocker une mémoire simple (`palace_store`)

Demandez à l'agent en langage naturel :

> « Stocke dans palazzo que **nous avons décidé d'utiliser Postgres plutôt
> que MongoDB pour le service commande**. C'est une décision technique du
> projet `commande-api`. »

L'agent appellera `palace_store` avec les bons champs automatiquement. Il peut
aussi vérifier qu'une mémoire similaire n'existe pas déjà (`palace_check_duplicate`).

### Corriger / remplacer une mémoire (`palace_supersede`)

Les faits évoluent. Quand une ancienne mémoire devient fausse, **on ne la
supprime pas** — on la remplace, et l'historique est préservé :

> « La mémoire #427 dit qu'on utilise Postgres. On a migré vers SQLite le mois
> dernier pour simplifier le déploiement. Utilise `palace_supersede` pour
> corriger ça. »

L'ancienne entrée est marquée `valid_until=<maintenant>` avec la raison, et la
nouvelle entrée pointe vers l'ancienne via `supersedes`. Par défaut, les
recherches ne verront que la version courante — mais l'historique reste
consultable.

### Bonnes pratiques

- **Soyez verbatim.** Ne résumez pas, ne paraphrasez pas. Le moteur d'embeddings
  fonctionne bien mieux sur des phrases complètes et factuelles.
- **Un fait = une mémoire.** Évitez les listes fourre-tout. Préférez plusieurs
  `palace_store` atomiques.
- **Indiquez toujours un `room`.** Un `room` vide rend la recherche par projet
  inutile.
- **Préférez `decisions` / `discoveries` à `facts` quand c'est pertinent** —
  le champ `hall` sert à filtrer plus tard (« montre-moi toutes les décisions
  du projet X »).

---

## 3. Comment retrouver (read)

### Recherche sémantique (`palace_find`)

La recherche par défaut est en langage naturel, sur tout le corpus :

> « Cherche dans palazzo : **pourquoi on utilise Postgres pour commande-api** »

Filtres optionnels disponibles :

| Filtre | Rôle |
|---|---|
| `wing`, `category`, `room`, `hall` | Restreindre à une facette précise |
| `since`, `until` (RFC3339) | Borne temporelle — ex. `since="2026-04-01T00:00:00Z"` |
| `recency_half_life_days` | Booster les mémoires récentes (ex. `365` = demi-vie d'un an) |
| `include_superseded` | Voir aussi les mémoires corrigées (archéologie) — `false` par défaut |
| `limit` | Nombre de résultats (défaut 5, max 20) |

Exemple complet :

> « Trouve les décisions techniques du projet `commande-api` prises depuis
> janvier, en ne gardant que les plus récentes. »

L'agent traduira ça en :

```json
{
  "query": "décisions techniques commande-api",
  "wing": "projects",
  "room": "commande-api",
  "hall": "decisions",
  "since": "2026-01-01T00:00:00Z",
  "recency_half_life_days": 90
}
```

### Récupérer par ID (`palace_recall`)

Quand vous connaissez déjà l'ID d'une mémoire (par exemple pour la citer ou la
corriger), pas besoin de refaire une recherche vectorielle :

> « Récupère les mémoires 427, 512 et 1776847707723. »

Retourne le contenu verbatim + les métadonnées temporelles (`valid_from`,
`valid_until`, `superseded_by`, `supersedes`).

### Vue d'ensemble (`palace_status`, `palace_taxonomy`)

Pour savoir « qu'est-ce que le palais contient ? » avant de chercher :

> « Donne-moi le `palace_status`. »

Retourne le nombre total de points + ventilation par *wing*, *hall* et
*category*. Utile en début de session pour calibrer les requêtes.

`palace_taxonomy` est similaire mais inclut aussi la liste des `room` (projets
et sujets) déjà connus.

---

## Antisèche

| Besoin | Outil | Prompt naturel |
|---|---|---|
| Stocker un fait | `palace_store` | « Stocke que… » |
| Corriger un fait ancien | `palace_supersede` | « La mémoire #X est périmée, remplace-la par… » |
| Éviter les doublons | `palace_check_duplicate` | « Vérifie qu'on n'a pas déjà ça avant de stocker » |
| Chercher par sens | `palace_find` | « Cherche dans palazzo… » |
| Récupérer par ID | `palace_recall` | « Récupère la mémoire #X » |
| Voir ce qu'il y a | `palace_status` / `palace_taxonomy` | « Montre-moi l'état du palais » |

---

## Support

- Code source : https://github.com/calibrae/palazzo
- Problème technique (serveur down, port non accessible) : voir l'équipe infra.
- Question sur l'usage : ce guide, puis demander directement à l'agent — il
  connaît le schéma et les outils.
