import psycopg2 
from psycopg2 import sql
import base64
from PIL import Image

def create_connection(db_params): 
	# Connect to the database 
	# using the psycopg2 adapter. 
	# Pass your database name ,# username , password ,  
	# hostname and port number 
	conn = psycopg2.connect(dbname=db_params['dbname'],
							user=db_params['user'],
							password=db_params['password'],
							host=db_params['host'],							
							port=db_params['port'])

	# Get the cursor object from the connection object 
	curr = conn.cursor() 
	return conn, curr  

def close_connection(connection, cursor):
	cursor.close()
	connection.close()


def create_image_table(conn, curr): 
	try: 
		# CREATE image table if it is non existant
		curr.execute("CREATE TABLE IF NOT EXISTS images(imageID SERIAL PRIMARY KEY, isGood BOOLEAN, imageData BYTEA)") 
		conn.commit()

	except(Exception, psycopg2.Error) as error: 
		print("Error while creating image table: ", error) 
		conn.close()


def upload_image_to_DB(conn, curr, image_file, is_good):
	try: 
		image_data = image_file.read()
		sql = "INSERT INTO images (isGood, imageData) VALUES (%(is_good)s, %(image_data)s);"
		data_insert_image = {
			'is_good': is_good,
			'image_data': psycopg2.Binary(image_data),
		}
		# Insert the image data into the database
		curr.execute(sql, data_insert_image)
		conn.commit()

	except(Exception, psycopg2.Error) as error: 
		print("could not upload the image: ", error) 
		conn.close()

