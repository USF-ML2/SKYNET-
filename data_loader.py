__author__ = "Harry"

import psycopg2, psycopg2.extras
import csv

DB_DSN = "host=localhost dbname=ml2 user=Harry password=XXX"


def transform_data():
    new_data = list()
    with open("final.csv", "rb") as f:
        reader = csv.reader(f, delimiter=",")
        # for x, y, driver_id, trip_id, step in reader:
        for line in reader:
        	if line != []:
	        	try:
	        	    row = (line[2], line[3], line[4], line[0], line[1])
	        	except ValueError as e:
	        		pass
        	new_data.append(row)
    return new_data


def drop_table():
    try:
        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        cur.execute("DROP TABLE trips")
        conn.commit()
        print "DROP TABLE"
    except psycopg2.Error as e:
        print e
    else:
        cur.close()
        conn.close()


def create_trips():
    try:
        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        cur.execute("CREATE TABLE trips("
                    "    x FLOAT"
                    "    , y FLOAT"
                    "    , driver_id INT"
                    "    , trip_id INT"
                    "    , step INT)")
        conn.commit()
        print "CREATE TABLE"
    except psycopg2.Error as e:
        print e
    else:
        cur.close()
        conn.close()


def insert_data(data):
    try:
        sql = "INSERT INTO trips VALUES (%s, %s, %s, %s, %s)"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.executemany(sql, data)  # NOTE executemany() as opposed to execute()
        conn.commit()
    except psycopg2.Error as e:
        print e.message
    else:
        cur.close()
        conn.close()

if __name__ == "__main__":
    # DATA = transform_data()
    # print DATA
    drop_table()
    create_trips()
    # insert_data(DATA)