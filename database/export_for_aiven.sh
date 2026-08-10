#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
set -a
source .env
set +a

MYSQLDUMP_BIN="mysqldump"
if ! command -v mysqldump >/dev/null 2>&1; then
    if [ -x "/usr/local/mysql/bin/mysqldump" ]; then
        MYSQLDUMP_BIN="/usr/local/mysql/bin/mysqldump"
    else
        echo "mysqldump not found on PATH or at /usr/local/mysql/bin/mysqldump" >&2
        exit 1
    fi
fi

OUTPUT="database/full_dump.sql"

# --no-tablespaces and --column-statistics=0 avoid needing the PROCESS
# privilege and querying information_schema.column_statistics, both of
# which a restricted managed-MySQL user (like Aiven's free tier) may not
# have. Views are dumped with DEFINER=`user`@`host` by default, which
# fails to import under a different Aiven username, so that clause is
# stripped — the importing user becomes the definer instead. No
# --databases flag, since Aiven provisions its own database name; import
# this into whatever database you point mysql at.
"$MYSQLDUMP_BIN" \
    -u root -p"${DB_PASSWORD}" \
    --single-transaction \
    --no-tablespaces \
    --column-statistics=0 \
    --set-gtid-purged=OFF \
    --default-character-set=utf8mb4 \
    credit_risk_platform \
    | sed -E 's/DEFINER=`[^`]*`@`[^`]*`//g' \
    > "$OUTPUT"

echo "Wrote $OUTPUT ($(du -h "$OUTPUT" | cut -f1), $(wc -l < "$OUTPUT") lines)"
echo
echo "Import into Aiven with:"
echo "  mysql -h <aiven-host> -P <aiven-port> -u <aiven-user> -p --ssl-mode=REQUIRED <aiven-database> < $OUTPUT"
