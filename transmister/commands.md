# Supervisor

sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl status all
sudo supervisorctl status toromate
sudo supervisorctl restart toromate

sudo supervisorctl restart all

# Nginx

sudo service nginx restart

# REDIS

redis-server
sudo systemctl restart redis.service
sudo /etc/init.d/redis-server restart
redis-server -v # Version check

# Celery
celery -A toromate beat -l info

celery -A toromate beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler
celery -A toromate worker -l info

# NPM

pm2 ls
pm2 start (from within the Vue project directory)
pm2 restart

# Languages

django-admin makemessages -l he -i venv_bingo
django-admin makemessages -l es -i venv_bingo
django-admin compilemessages

# Ngrok or LocalTunnel
./ngrok http 8030
lt --port 8030