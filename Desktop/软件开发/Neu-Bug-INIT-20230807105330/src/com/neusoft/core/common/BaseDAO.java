package com.neusoft.core.common;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.beanutils.BeanUtils;
import org.myframework.jdbc.sqltemplate.impl.VelocitySqlTemplate;
import org.myframework.util.StringUtil;
import org.myframework.util.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.dao.DataAccessException;
import org.springframework.dao.EmptyResultDataAccessException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.namedparam.BeanPropertySqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.core.namedparam.SqlParameterSource;

import com.neusoft.core.common.exception.DAOException;

/**
 * @ClassName: BaseDAO
 * @Description: 基础DAO类
 */
public class BaseDAO extends BaseObject {

	/**
	 * @Fields jdbcTemplate : SpringMVC框架注入的参数JdbcTemplate 对象
	 */
	@Autowired
	private JdbcTemplate jdbcTemplate;

	/**
	 * @Fields npjt : SpringMVC框架注入的参数NamedParameterJdbcTemplate 对象
	 */
	@Autowired
	private NamedParameterJdbcTemplate npjt;

	
	/**
	 * @Title: executeSQL
	 * @Description: 执行单条sql语句，如create table或者drop table
	 * @param sql
	 *            要执行的sql语句
	 * @throws DAOException
	 */
	protected void executeSQL(String sql) throws DAOException {
		try {
			this.jdbcTemplate.execute(sql);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: update
	 * @Description: 执行单条insert、update、delete语句
	 * @param sql
	 *            要执行的sql语句
	 * @return 执行语句影响数据的条数
	 * @throws DAOException
	 */
	protected int update(String sql) throws DAOException {
		try {
			return this.jdbcTemplate.update(sql);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: update
	 * @Description: 执行单条insert、update、delete语句
	 * @param sql
	 *            要执行的sql语句
	 * @param args
	 *            参数
	 * @return 执行语句影响数据的条数
	 * @throws DAOException
	 */
	protected int update(String sql, Object... args) throws DAOException {
		try {
			return this.jdbcTemplate.update(sql, args);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: update
	 * @Description: 执行单条insert、update、delete语句
	 * @param sql
	 *            要执行的sql语句
	 * @param bean
	 *            语句中的参数,从Bean中获取
	 * @return 执行语句影响数据的条数
	 */
	protected int update(String sql, Object bean) throws DAOException {
		try {
			SqlParameterSource namedParameters = new BeanPropertySqlParameterSource(
					bean);
			return this.npjt.update(sql, namedParameters);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: updateBatch
	 * @Description: 执行批量insert、update、delete语句
	 * @param sql
	 *            要执行的sql语句
	 * @return 执行语句影响数据的条数
	 */
	protected int[] updateBatch(String[] sql) throws DAOException {
		try {
			return this.jdbcTemplate.batchUpdate(sql);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: batchUpdate
	 * @Description: 执行批量insert、update、delete语句
	 * @param sql
	 *            要执行的sql语句
	 * @param args
	 *            参数
	 * @return int[] 执行语句影响数据的条数
	 */
	protected int[] batchUpdate(String sql, List<Object[]> args) {
		int[] r = jdbcTemplate.batchUpdate(sql, args);
		return r;
	}

	/**
	 * @Title: batchUpdate
	 * @Description: 执行批量insert、update、delete语句
	 * @param sql
	 *            要执行的sql语句
	 * @param args
	 *            参数
	 * @return int[] 执行语句影响数据的条数
	 */
	protected int[] batchUpdate(String sql, Map<String, Object>[] args) {
		int[] r = npjt.batchUpdate(sql, args);
		return r;
	}

	/**
	 * @Title: queryForObject
	 * @Description: 执行select语句，获取获取一行数据
	 * @param sql
	 *            要执行的语句
	 * @param requiredType
	 *            返回对象类型
	 * @return 单个数据对象
	 * @throws DAOException
	 */
	protected <T> T queryForObject(String sql, Class<T> requiredType)
			throws DAOException {
		try {
			return this.jdbcTemplate.queryForObject(sql, requiredType);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForObject
	 * @Description: 执行select语句，获取获取一行数据
	 * @param sql
	 *            要执行的语句
	 * @param requiredType
	 *            返回对象类型
	 * @param args
	 *            参数
	 * @return 单个数据对象
	 * @throws DAOException
	 */
	protected <T> T queryForObject(String sql, Class<T> requiredType,
			Object... args) throws DAOException {
		try {
			return this.jdbcTemplate.queryForObject(sql, requiredType, args);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForBean
	 * @Description: 执行select语句，获取获取一行数据
	 * @param sql
	 *            要执行的语句
	 * @param requiredType
	 *            返回对象类型
	 * @return Bean数据对象
	 * @throws DAOException
	 */
	@SuppressWarnings({ "unchecked", "rawtypes" })
	protected <T> T queryForBean(String sql, Class<T> requiredType)
			throws DAOException {
		try {
			Map m = this.jdbcTemplate.queryForMap(sql);
			Class<?> cls = Class.forName(requiredType.getName());// 取得Class对象
			Object obj = cls.newInstance();
			T t = (T) obj;
			BeanUtils.copyProperties(t, m);
			return t;
		} catch (EmptyResultDataAccessException ede) {
			log.debug(ede.getMessage());
			return null;
		} catch (Exception e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForBean
	 * @Description: 执行select语句，获取获取一行数据
	 * @param sql
	 *            要执行的语句
	 * @param requiredType
	 *            返回对象类型
	 * @param args
	 *            参数
	 * @return Bean数据对象
	 * @throws DAOException
	 */
	@SuppressWarnings({ "unchecked", "rawtypes" })
	protected <T> T queryForBean(String sql, Class<T> requiredType,
			Object... args) throws DAOException {
		try {
			Map m = this.jdbcTemplate.queryForMap(sql, args);
			Class<?> cls = Class.forName(requiredType.getName());// 取得Class对象
			Object obj = cls.newInstance();
			T t = (T) obj;
			BeanUtils.copyProperties(t, m);
			return t;
		} catch (EmptyResultDataAccessException ede) {
			log.debug(ede.getMessage());
			return null;
		} catch (Exception e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForBeanByBean
	 * @Description: 执行select语句，获取获取一行数据
	 * @param sql
	 *            要执行的语句
	 * @param requiredType
	 *            返回对象类型
	 * @param bean
	 *            bean类型参数
	 * @return 单个数据对象
	 * @throws DAOException
	 */
	@SuppressWarnings({ "unchecked", "rawtypes" })
	protected <T> T queryForBeanByBean(String sql, Class<T> requiredType,
			Object bean) throws DAOException {
		try {
			SqlParameterSource namedParameters = new BeanPropertySqlParameterSource(
					bean);
			Map m = this.npjt.queryForMap(sql, namedParameters);
			Class<?> cls = Class.forName(requiredType.getName());// 取得Class对象
			Object obj = cls.newInstance();
			T t = (T) obj;
			BeanUtils.copyProperties(t, m);
			return t;
		} catch (EmptyResultDataAccessException ede) {
			log.debug(ede.getMessage());
			return null;
		} catch (Exception e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForBeanByBeanMap
	 * @Description: 执行select语句，获取获取一行数据
	 * @param sql
	 *            要执行的语句
	 * @param requiredType
	 *            返回对象类型
	 * @param paramMap
	 *            map类型参数
	 * @return 单个数据对象
	 * @throws DAOException
	 *             T
	 */
	@SuppressWarnings({ "unchecked", "rawtypes" })
	protected <T> T queryForBeanByBeanMap(String sql, Class<T> requiredType,
			Map<String, ?> paramMap) throws DAOException {
		try {
			Map m = this.npjt.queryForMap(sql, paramMap);
			Class<?> cls = Class.forName(requiredType.getName());// 取得Class对象
			Object obj = cls.newInstance();
			T t = (T) obj;
			BeanUtils.copyProperties(t, m);
			return t;
		} catch (EmptyResultDataAccessException ede) {
			log.debug(ede.getMessage());
			return null;
		} catch (Exception e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForMap
	 * @Description: 执行select语句，获取获取一行数据
	 * @param sql
	 *            要执行的sql语句
	 * @return map对象，key就是是列的名称
	 */
	protected Map<String, Object> queryForMap(String sql) throws DAOException {
		try {
			return this.jdbcTemplate.queryForMap(sql);
		} catch (EmptyResultDataAccessException ede) {
			log.debug(ede.getMessage());
			return null;
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForMap
	 * @Description: 执行select语句，获取获取一行数据
	 * @param sql
	 *            要执行的sql语句
	 * @param args
	 *            sql语句中的参数
	 * @return map对象，key就是是列的名称
	 */
	protected Map<String, Object> queryForMap(String sql, Object... args)
			throws DAOException {
		try {
			return this.jdbcTemplate.queryForMap(sql, args);
		} catch (EmptyResultDataAccessException ede) {
			log.debug(ede.getMessage());
			return null;
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForMap
	 * @Description: 执行select语句，获取获取一行数据
	 * @param sql
	 *            要执行的sql语句
	 * @param bean
	 *            bean对象
	 * @return Map<String, Object>
	 * @throws DAOException
	 */
	protected Map<String, Object> queryForMapByBean(String sql, Object bean)
			throws DAOException {
		try {
			SqlParameterSource namedParameters = new BeanPropertySqlParameterSource(
					bean);
			return this.npjt.queryForMap(sql, namedParameters);
		} catch (EmptyResultDataAccessException ede) {
			log.debug(ede.getMessage());
			return null;
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForMapByBeanMap
	 * @Description: 执行select语句，获取获取一行数据
	 * @param sql
	 *            要执行的sql语句
	 * @param paramMap
	 *            map类型参数
	 * @return Map<String, Object>
	 * @throws DAOException
	 */
	protected Map<String, Object> queryForMapByBeanMap(String sql,
			Map<String, ?> paramMap) throws DAOException {
		try {
			return this.npjt.queryForMap(sql, paramMap);
		} catch (EmptyResultDataAccessException ede) {
			log.debug(ede.getMessage());
			return null;
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForString
	 * @Description: 执行select语句，获取一个字符串
	 * @param sql
	 *            要执行的语句
	 * @return 字符串
	 */
	protected String queryForString(String sql) throws DAOException {
		try {
			return this.jdbcTemplate.queryForObject(sql, String.class);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForString
	 * @Description: 执行select语句，获取一个字符串
	 * @param sql
	 *            要执行的语句
	 * @param args
	 *            参数
	 * @return 字符串
	 * @throws DAOException
	 */
	protected String queryString(String sql, Object... args)
			throws DAOException {
		try {
			return this.jdbcTemplate.queryForObject(sql, String.class, args);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForInteger
	 * @Description: 执行select语句，获取一个int型数据
	 * 
	 * @param sql
	 *            要执行的语句
	 * @return 数字
	 */
	protected int queryForInteger(String sql) throws DAOException {
		try {
			return this.jdbcTemplate.queryForObject(sql, Integer.class);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForInteger
	 * @Description: 执行select语句，获取一个int型数据
	 * @param sql
	 *            要执行的语句
	 * @param args
	 *            参数
	 * @return 数字
	 * @throws DAOException
	 */
	protected int queryInteger(String sql, Object... args) throws DAOException {
		try {
			return this.jdbcTemplate.queryForObject(sql, Integer.class, args);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForInteger
	 * @Description: 执行select语句，获取一个int型数据
	 * @param sql
	 *            要执行的语句
	 * @param params
	 *            参数
	 * @return int
	 * @throws DAOException
	 */
	protected int queryInteger(String sql, Map<String, ?> params)
			throws DAOException {
		try {
			return this.npjt.queryForObject(sql, params, Integer.class);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForList
	 * @Description: 执行select语句，获取列表数据
	 * @param sql
	 *            要执行的语句
	 * @param elementType
	 *            对象类型
	 * @return 数据列表
	 */
	protected <T> List<T> queryForList(String sql, Class<T> elementType)
			throws DAOException {
		try {
			return this.jdbcTemplate.queryForList(sql, elementType);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForList
	 * @Description: 执行select语句，获取列表数据
	 * @param sql
	 *            要执行的语句
	 * @param elementType
	 *            对象类型
	 * @param args
	 *            语句中的参数
	 * @return 数据列表
	 */
	protected <T> List<T> queryForList(String sql, Class<T> elementType,
			Object... args) throws DAOException {
		try {
			return this.jdbcTemplate.queryForList(sql, elementType, args);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForBeanList
	 * @Description: 执行select语句，获取列表数据
	 * @param sql
	 *            要执行的语句
	 * @param requiredType
	 *            对象类型
	 * @return List<Bean>数据列表
	 */
	@SuppressWarnings("unchecked")
	protected <T> List<T> queryForBeanList(String sql, Class<T> requiredType)
			throws DAOException {
		try {
			List<T> resultList = new ArrayList<T>();
			Class<?> cls = Class.forName(requiredType.getName());// 取得Class对象

			List<Map<String, Object>> list = this.jdbcTemplate
					.queryForList(sql);
			for (Map<String, Object> m : list) {
				T t = (T) cls.newInstance();
				BeanUtils.copyProperties(t, m);
				resultList.add(t);
			}
			return resultList;
		} catch (Exception e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForBeanList
	 * @Description: 执行select语句，获取列表数据
	 * @param sql
	 *            要执行的语句
	 * @param requiredType
	 *            对象类型
	 * @param args
	 *            语句中的参数
	 * @return 数据列表
	 */
	@SuppressWarnings("unchecked")
	protected <T> List<T> queryForBeanList(String sql, Class<T> requiredType,
			Object... args) throws DAOException {
		try {
			List<T> resultList = new ArrayList<T>();
			Class<?> cls = Class.forName(requiredType.getName());// 取得Class对象

			List<Map<String, Object>> list = this.jdbcTemplate.queryForList(
					sql, args);
			for (Map<String, Object> m : list) {
				T t = (T) cls.newInstance();
				BeanUtils.copyProperties(t, m);
				resultList.add(t);
			}
			return resultList;
		} catch (Exception e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForBeanListByBean
	 * @Description: 执行select语句，获取列表数据
	 * @param sql
	 *            要执行的语句
	 * @param requiredType
	 *            对象类型
	 * @param bean
	 *            语句中的参数,从Bean中获取
	 * @return List<T>
	 */
	@SuppressWarnings("unchecked")
	protected <T> List<T> queryForBeanListByBean(String sql,
			Class<T> requiredType, Object bean) {
		try {
			List<T> resultList = new ArrayList<T>();
			Class<?> cls = Class.forName(requiredType.getName());// 取得Class对象
			SqlParameterSource namedParameters = new BeanPropertySqlParameterSource(
					bean);
			List<Map<String, Object>> list = this.npjt.queryForList(sql,
					namedParameters);
			for (Map<String, Object> m : list) {
				T t = (T) cls.newInstance();
				BeanUtils.copyProperties(t, m);
				resultList.add(t);
			}
			return resultList;
		} catch (Exception e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForBeanListByBeanMap
	 * @Description: 执行select语句，获取列表数据
	 * @param sql
	 *            要执行的语句
	 * @param requiredType
	 *            对象类型
	 * @param paramMap
	 *            语句中的参数,从Map中获取
	 * @return 数据列表
	 */
	@SuppressWarnings("unchecked")
	protected <T> List<T> queryForBeanListByBeanMap(String sql,
			Class<T> requiredType, Map<String, ?> paramMap) {
		try {
			List<T> resultList = new ArrayList<T>();
			Class<?> cls = Class.forName(requiredType.getName());// 取得Class对象
			List<Map<String, Object>> list = this.npjt.queryForList(sql,
					paramMap);
			for (Map<String, Object> m : list) {
				T t = (T) cls.newInstance();
				BeanUtils.copyProperties(t, m);
				resultList.add(t);
			}
			return resultList;
		} catch (Exception e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForList
	 * @Description: 执行select语句，获取列表数据
	 * @param sql
	 *            要执行的语句
	 * @return List列表数据
	 */
	protected List<Map<String, Object>> queryForList(String sql) {
		try {
			return this.jdbcTemplate.queryForList(sql);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForList
	 * @Description: 执行select语句，获取列表数据
	 * @param sql
	 *            sql语句
	 * @param args
	 *            参数
	 * @return List列表数据
	 */
	protected List<Map<String, Object>> queryForList(String sql, Object... args) {
		try {
			return this.jdbcTemplate.queryForList(sql, args);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForList
	 * @Description: 执行select语句，获取列表数据
	 * @param sql
	 *            sql语句
	 * @param bean
	 *            bean类型参数
	 * @return List<Map<String,Object>>
	 */
	protected List<Map<String, Object>> queryForListByBean(String sql,
			Object bean) {
		try {
			SqlParameterSource namedParameters = new BeanPropertySqlParameterSource(
					bean);
			return this.npjt.queryForList(sql, namedParameters);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: queryForListByBeanMap
	 * @Description: 执行select语句，获取列表数据
	 * @param sql
	 *            sql语句
	 * @param paramMap
	 *            map类型参数
	 * @return List<Map<String,Object>>
	 */
	protected List<Map<String, Object>> queryForListByBeanMap(String sql,
			Map<String, ?> paramMap) {
		try {
			return this.npjt.queryForList(sql, paramMap);
		} catch (DataAccessException e) {
			throw new DAOException(e.getMessage(), e);
		}
	}

	/**
	 * @Title: getLimitSQL
	 * @Description: 获取LIMIT语句串，主要用于Mysql分页查询
	 * @param index
	 *            开始行索引
	 * @param count
	 *            结果记录的条数
	 * @return LIMIT语句串
	 */
	protected String getLimitSQL(int index, int count) {
		return new StringBuffer().append(" LIMIT ").append(index).append(",")
				.append(count).toString();
	}

	/**
	 * @Title: getCountSQL
	 * @Description: 获取count语句串，主要用于Mysql分页查询
	 * @param sql
	 *            执行翻页sql
	 * @return String
	 */
	protected String getCountSQL(String sql) {
		return new StringBuffer().append(" select count(1) from (").append(sql)
				.append(") _auto_count ").toString();
	}

	/**
	 * @Title: getCount
	 * @Description: 获得数据行数
	 * @param srcSql
	 *            执行翻页sql
	 * @param args
	 *            参数
	 * @return int 行数
	 */
	protected int getCount(String srcSql, Object... args) {
		return this.queryInteger(getCountSQL(srcSql), args);
	}

	/**
	 * @Title: getCount
	 * @Description: 获得数据行数
	 * @param srcSql
	 *            执行翻页sql
	 * @param paramMap
	 *            参数
	 * @return int 行数
	 */
	protected int getCount(String srcSql, Map<String, ?> paramMap) {
		return this.queryInteger(getCountSQL(srcSql), paramMap);
	}

	/**
	 * @Title: getPageData
	 * @Description: 获取分页列表数据
	 * @param srcSql
	 *            执行翻页sql
	 * @param pageIndex
	 *            页的开始行数
	 * @param pageSize
	 *            页的总行数
	 * @param requiredType
	 *            返回List中的元素类型
	 * @param args
	 *            参数
	 * @return 分页列表数据
	 * @throws DAOException
	 */
	protected <T> List<T> getPageData(String srcSql, Class<T> requiredType,
			int pageIndex, int pageSize, Object... args) throws DAOException {
		StringBuffer pagingSql = new StringBuffer();
		pagingSql.append(srcSql);
		pagingSql.append(this.getLimitSQL(pageIndex, pageSize));
		return this.queryForBeanList(pagingSql.toString(), requiredType, args);
	}

	/**
	 * @Title: getPageData
	 * @Description: 获取分页列表数据
	 * @param srcSql
	 *            执行的sql语句
	 * @param requiredType
	 *            返回List中的元素类型
	 * @param pageIndex
	 *            页的开始行数
	 * @param pageSize
	 *            页的总行数
	 * @param paramMap
	 *            参数Map
	 * @return 分页列表数据
	 * @throws DAOException
	 */
	protected <T> List<T> getPageData(String srcSql, Class<T> requiredType,
			int pageIndex, int pageSize, Map<String, ?> paramMap)
			throws DAOException {
		StringBuffer pagingSql = new StringBuffer();
		pagingSql.append(srcSql);
		pagingSql.append(this.getLimitSQL(pageIndex, pageSize));
		return this.queryForBeanListByBeanMap(pagingSql.toString(),
				requiredType, paramMap);
	}

	

	
	/**
	 * @Title: sqlConvert
	 * @Description: 解析sql
	 * @param sql
	 *            执行的sql语句
	 * @param qryMap
	 *            参数
	 * @param list
	 * @return 解析sql
	 */
	@SuppressWarnings("rawtypes")
	private String sqlConvert(String sql, Map qryMap, List<Object> list) {
		// replace #param to value
		if (qryMap != null && !qryMap.isEmpty()) {
			List<String> params = getReplaceParameterList(sql);
			for (String param : params) {
				sql = StringUtils.replaceOnce(sql, "#" + param,
						StringUtil.asString(qryMap.get(param)));
			}
		}
		return sql;
	}

	/**
	 * @Title: getReplaceParameterList
	 * @Description: 获取模板sql参数
	 * @param sql
	 *            sql语句
	 * @return List
	 */
	private List<String> getReplaceParameterList(String sql) {
		String regex = "\\#(\\w+)";
		Pattern pattern = Pattern.compile(regex);
		Matcher matcher = pattern.matcher(sql);
		List<String> params = new ArrayList<String>();
		while (matcher.find())
			params.add(matcher.group(1));
		return params;
	}
}
