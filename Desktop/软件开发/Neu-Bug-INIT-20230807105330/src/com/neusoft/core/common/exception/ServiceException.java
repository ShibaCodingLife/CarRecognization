package com.neusoft.core.common.exception;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * @ClassName: ServiceException
 * @Description: Service层Exception简单封装,用于异常信息向Controller抛出
 */
public class ServiceException extends RuntimeException {

	/**
	 * @Fields serialVersionUID : 序列化编号
	 */
	private static final long serialVersionUID = 3517241863691434831L;

	/**
	 * @Fields log : 日志对象
	 */
	protected final Log log = LogFactory.getLog(getClass());

	/**
	 * @param message
	 *            异常信息
	 */
	public ServiceException(String message) {
		super(message);
		log.error(message);
	}

	/**
	 * @param message
	 *            异常信息
	 * @param throwable
	 *            异常对象
	 */
	public ServiceException(String message, Throwable throwable) {
		super(message, throwable);
		log.error(message + " :Throwable[" + throwable.getMessage() + "]");
	}
}
