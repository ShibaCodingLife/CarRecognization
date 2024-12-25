package com.neusoft.core.common.exception;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * @ClassName: DAOException
 * @Description: DAO层Exception简单封装,用于异常信息向Service抛出
 */
public class DAOException extends RuntimeException {

	/**
	 * @Fields serialVersionUID : 序列化ID
	 */
	private static final long serialVersionUID = 8006839181281782460L;

	/**
	 * @Fields log : 日志对象
	 */
	protected final Log log = LogFactory.getLog(getClass());

	/**
	 * @param message
	 *            异常信息
	 */
	public DAOException(String message) {
		super(message);
		log.error(message);
	}

	/**
	 * @param message
	 *            异常信息
	 * @param throwable
	 *            异常对象
	 */
	public DAOException(String message, Throwable throwable) {
		super(message, throwable);
		log.error(message + " :Throwable[" + throwable.getMessage() + "]");
	}
}
