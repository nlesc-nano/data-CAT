
## Deploy mongodb in as a Docker container

First you need to install [Docker](https://docs.docker.com/install/) in the host, then use the following steps:

### Get the image

```bash
docker pull mongo
```

### Run the container
```bash
docker run -d -p 27017-27019:27017-27019 -v <path/db/local/host>:/data/db --name mongodb mongo mongod --auth
```

### Enter to the container
```bash
docker exec -it mongodb bash
```

### Connect to local mongo
```
mongo
```

### Create db administrator
```
use admin
db.createUser( { user: "useradmin", pwd: "passwd", roles: [ { role: "userAdminAnyDatabase", db: "admin" } ] } )
```

### Exit the mongo shell and the container
```
exit
exit
```

### Conect with the admin user
mongo -u useradmin -p "passwd" <HOSTNAME/IP> --authenticationDatabase "admin"
