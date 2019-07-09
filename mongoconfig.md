
## Deploy mongodb in as a Docker container

First you need to install [Docker](https://docs.docker.com/install/) in the host, then use the following steps:

### Get the image

```bash
docker pull mongo
```

### Run the container
```bash
docker run -d -p 27017-27019:27017-27019 -v <path/db/local/host>:/data/db --name mongodb -e MONGO_INITDB_ROOT_USERNAME=admin -e MONGO_INITDB_ROOT_PASSWORD=strong_password  mongo
```

### Enter to the container
```bash
docker exec -it mongodb bash
```

### Connect mongo instance in the container
```
mongo -u admin -p <previous password>
```

### Create CAT Database user
```
use admin
db.createUser( { user: "cat", pwd: "passwd", roles: [ { role: "dbOwner", db: "cat_database" } ] } )
```
### Exit the mongo shell and the container
```
exit
exit
```

### Conect with the admin user
mongo -u useradmin -p "passwd" <HOSTNAME/IP> --authenticationDatabase "admin"
