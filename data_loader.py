import psycopg2, psycopg2.extras
import json
from pprint import pprint

DB_DSN = "host=localhost dbname=ml2 user=Harry password=HOR99d31991"


def transform_data():
    new_data = list()
    with open("1.csv", "rb") as f:
        for line in f:
            l = line.read()
    for entry in l:
    	tuple()
            except Exception as e:
                pass
    return new_data


def drop_table():
    try:
        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        cur.execute("DROP TABLE results")
        conn.commit()
        print "DROP TABLE"
    except psycopg2.Error as e:
        print e
    else:
        cur.close()
        conn.close()


def create_horse_table():
    try:
        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        cur.execute("CREATE TABLE results("
                    "    uid int"
                    "    , datetime TIMESTAMP"
                    "    , venue TEXT"
                    "    , dist FLOAT"
                    "    , grade TEXT"
                    "    , ground TEXT"
                    "    , name TEXT"
                    "    , jockey TEXT"
                    "    , trainer TEXT"
                    "    , place TEXT"
                    "    , sp FLOAT"
                    "    , weight FLOAT"
                    "    , age INTEGER"
                    "    , country TEXT"
                    "    , o_rating TEXT"
                    "    , topspeed TEXT"
                    "    , rpr TEXT)")
        conn.commit()
        print "CREATE TABLE"
    except psycopg2.Error as e:
        print e
    else:
        cur.close()
        conn.close()


def insert_data(data):
    try:
        sql = "INSERT INTO results VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
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
    DATA = transform_data()
    # print DATA
    drop_table()
    create_horse_table()
    insert_data(DATA)