#!/bin/bash
# Vox Probabilis · daily SQLite backup (DEPLOY.md §11)
#
# Install:
#   sudo install -m 755 /opt/voxprobabilis/backend/deploy/voxprobabilis-backup.sh \
#                       /etc/cron.daily/voxprobabilis-backup
#   sudo /etc/cron.daily/voxprobabilis-backup        # one dry run to verify
#
# What it does:
#   - sqlite3 .backup (WAL-aware; safer than `cp` against a live DB)
#   - 14-day retention (older snapshots deleted)
#   - gzip snapshots older than 1 day to save space
#
# Off-VPS replication is deferred to v0.2; this protects against
# accidental deletion or DB corruption (the 90% real-world risk).

set -euo pipefail

BACKUP_DIR=/var/backups/voxprobabilis
DB=/var/lib/voxprobabilis/vox.db
DATE=$(date -u +%Y-%m-%d)

mkdir -p "$BACKUP_DIR"

# SQLite-aware backup — handles the WAL correctly, unlike a plain cp
# (which can capture a torn page if a write commits mid-copy).
sqlite3 "$DB" ".backup '$BACKUP_DIR/vox-$DATE.db'"

# Retention: 14 days
find "$BACKUP_DIR" -name 'vox-*.db'    -mtime +14 -delete
find "$BACKUP_DIR" -name 'vox-*.db.gz' -mtime +14 -delete

# Compress backups older than 1 day to save space.
find "$BACKUP_DIR" -name 'vox-*.db' -mtime +1 ! -name '*.gz' -exec gzip {} \;
