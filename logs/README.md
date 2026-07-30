# logs

All application topics share `lol/oracle-bets.log`. It rotates at 10 MiB and
keeps five backups by default, limiting retained application logs to roughly
60 MiB. Override only with `ORACLE_BETS_LOG_MAX_BYTES` and
`ORACLE_BETS_LOG_BACKUPS`. Log files are generated and ignored.
