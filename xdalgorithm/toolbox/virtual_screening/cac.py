import json
import os
import pymysql


def get_db():
    mysql_conf = json.load(open(os.path.expanduser("~/mysql.conf")))
    con = pymysql.connect(**mysql_conf)
    return con.cursor()


def __get_compound_by_slice(db_cur, table="commavailcmpd", p=0, s=10000):
    start = p
    end = start + s - 1
    sql = """SELECT *
             FROM {}
             where id between %s and %s  ORDER by id ASC  """.format(table)
    db_cur.execute(sql, (start, end))
    return db_cur.fetchall()


def __get_compound_ids(db_cur, table="commavailcmpd", formal_charge=None, HBA=None, HBD=None, rotB=None, MW=None,
                       logP=None, QED=None):
    sql = """SELECT *
                FROM {} where """.format(table)
    and_flag = False
    sql_values = []

    # formal_charge
    if formal_charge:
        sql += " formal_charge between %s and %s "
        sql_values.extend(formal_charge)
        and_flag = True

    # HBA
    if and_flag:
        sql += " and "
    if HBA:
        sql += " HBA between %s and %s "
        sql_values.extend(HBA)
        and_flag = True

    # HBD
    if and_flag:
        sql += " and "
    if HBD:
        sql += " HBD between %s and %s "
        sql_values.extend(HBD)
        and_flag = True

    # rotB
    if and_flag:
        sql += " and "
    if rotB:
        sql += " rotB between %s and %s "
        sql_values.extend(rotB)
        and_flag = True

    # MW
    if and_flag:
        sql += " and "
    if rotB:
        sql += " MW between %s and %s "
        sql_values.extend(MW)
        and_flag = True

    # logP
    if and_flag:
        sql += " and "
    if rotB:
        sql += " logP between %s and %s "
        sql_values.extend(logP)
        and_flag = True

    # QED
    if and_flag:
        sql += " and "
    if rotB:
        sql += " QED between %s and %s "
        sql_values.extend(QED)
    sql += " order by id ASC "
    db_cur.execute(sql, sql_values)
    return db_cur.fetchall()


def __get_compoud_by_ids(db_cur, ids, table='commavailcmpd'):
    sql = """SELECT *
                 FROM {} 
                 WHERE id IN %s order by id ASC """.format(table)
    db_cur.execute(sql, ids)
    return db_cur.fetchall()


def __get_compoud_vender_info_by_ids(db_cur, ids, table='commavailcmpdvendorinfo'):
    sql = """SELECT *
             FROM {}
             WHERE id IN %s order by id ASC, vendor_source ASC """.format(table)
    db_cur.execute(sql, ids)
    return db_cur.fetchall()


def __get_compoud_and_vendor_by_ids(db_cur, ids, table='commavailcmpd', table_vendor="commavailcmpdvendorinfo"):
    sql = """SELECT *
                 FROM {}
                 WHERE id IN %s order by id ASC UNION 
             SELECT *
                 FROM {}
                 WHERE id IN %s order by id ASC""".format(table, table_vendor)
    db_cur.execute(sql, (ids, ids))
    return db_cur.fetchall()


def __get_compoud_by_smiles(db_cur, smiles, table='commavailcmpd'):
    sql = """SELECT *
                 FROM {} 
                 WHERE smiles=%s """.format(table)
    db_cur.execute(sql, smiles.replace("\\", "\\\\"))
    return db_cur.fetchall()


def __get_compoud_and_vendor_by_smiles(db_cur, smiles, table='commavailcmpd', table_vendor="commavailcmpdvendorinfo"):
    sql = """SELECT * FROM {} INNER JOIN {}
             WHERE {}.smiles=%s and {}.id={}.id
             order by id ASC""".format(table, table_vendor, table, table, table_vendor)
    db_cur.execute(sql, smiles.replace("\\", "\\\\"))
    return db_cur.fetchall()
