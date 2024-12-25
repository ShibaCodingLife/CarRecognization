package com.neusoft.core.util;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.MultipartHttpServletRequest;
import org.springframework.web.multipart.commons.CommonsMultipartResolver;

/**
 * @ClassName: FileUtils
 * @Description: 文件上传工具类
 */
public class FileUtils {

	/**
	 * @Fields IMAGE_FILE_EXTENSION : 上传图片时允许的文件类型
	 */
	public static final String IMAGE_FILE_EXTENSION = ".jpg|.jpeg|.png|.gif|";

	/**
	 * @Fields EXCEL_FILE_EXTENSION : 上传Excel时允许的文件类型
	 */
	public static final String EXCEL_FILE_EXTENSION = ".xls|.xlsx|.csv|";

	/**
	 * @Fields MSEXCEL_FILE_EXTENSION : 上传MSEXCEL时允许的文件类型
	 */
	public static final String MSEXCEL_FILE_EXTENSION = ".xls|.xlsx|";

	/**
	 * @Fields MSWORD_FILE_EXTENSION : 上传MSWORD时允许的文件类型
	 */
	public static final String MSWORD_FILE_EXTENSION = ".doc|.docx|";

	/**
	 * @Fields MSPOWERPOINT_FILE_EXTENSION : 上传MSPOWERPOINT时允许的文件类型
	 */
	public static final String MSPOWERPOINT_FILE_EXTENSION = ".ppt|.pptx|";

	/**
	 * @Fields HTML_FILE_EXTENSION : 上传html时允许的文件类型
	 */
	public static final String HTML_FILE_EXTENSION = ".html|.htm|";

	/**
	 * @Fields TXT_FILE_EXTENSION : 上传txt时允许的文件类型
	 */
	public static final String TXT_FILE_EXTENSION = ".txt|";

	/**
	 * @Fields XML_FILE_EXTENSION : 上传xml时允许的文件类型
	 */
	public static final String XML_FILE_EXTENSION = ".xml|";

	/**
	 * @Fields PDF_FILE_EXTENSION : 上传pdf时允许的文件类型
	 */
	public static final String PDF_FILE_EXTENSION = ".pdf|";

	/**
	 * @Fields ATTACHMENT_EXTENSION : 可执行文件
	 */
	public static final String ATTACHMENT_EXTENSION = ".doc|.docx|.xls|.xlsx|.ppt|.pptx|.txt|.pdf|.zip|.rar|";

	/**
	 * @Fields FILE_SIZE_30M : 上传文件大小最大为30M
	 */
	public static final long FILE_SIZE_30M = 30;

	/**
	 * @Title: isImageFile
	 * @Description: 是否为图片
	 * @param imgFile
	 * @return boolean
	 */
	public static boolean isImageFile(MultipartFile imgFile) {
		// 提取上传文件的类型
		String fileName = imgFile.getOriginalFilename();
		// 获取上传文件类型的扩展名,先得到.的位置，再截取从.的下一个位置到文件的最后，最后得到扩展名 ，对扩展名进行小写转换
		String extName = fileName
				.substring(
						fileName.lastIndexOf(".") == -1 ? 0 : fileName
								.lastIndexOf(".")).toLowerCase();
		// 判断上传文件类型是否允许上传
		if (IMAGE_FILE_EXTENSION.toLowerCase().indexOf(extName.concat("|")) == -1) {
			return false;
		} else {
			return true;
		}
	}

	/**
	 * @Title: isExcelFile
	 * @Description: 是否为Excel文件
	 * @param imgFile
	 * @return boolean
	 */
	public static boolean isExcelFile(MultipartFile imgFile) {
		// 提取上传文件的类型
		String fileName = imgFile.getOriginalFilename();
		// 获取上传文件类型的扩展名,先得到.的位置，再截取从.的下一个位置到文件的最后，最后得到扩展名 ，对扩展名进行小写转换
		String extName = fileName
				.substring(
						fileName.lastIndexOf(".") == -1 ? 0 : fileName
								.lastIndexOf(".")).toLowerCase();
		// 判断上传文件类型是否允许上传
		if (EXCEL_FILE_EXTENSION.toLowerCase().indexOf(extName.concat("|")) == -1) {
			return false;
		} else {
			return true;
		}
	}

	/**
	 * @Title: isAttachment
	 * @Description: 是否为可执行文件
	 * @param imgFile
	 * @return boolean
	 */
	public static boolean isAttachment(MultipartFile imgFile) {
		if (imgFile != null) {
			// 提取上传文件的类型
			String fileName = imgFile.getOriginalFilename();
			// 获取上传文件类型的扩展名,先得到.的位置，再截取从.的下一个位置到文件的最后，最后得到扩展名 ，对扩展名进行小写转换
			String extName = fileName.substring(
					fileName.lastIndexOf(".") == -1 ? 0 : fileName
							.lastIndexOf(".")).toLowerCase();
			// 判断上传文件类型是否允许上传
			if (ATTACHMENT_EXTENSION.toLowerCase().indexOf(extName.concat("|")) == -1) {
				return false;
			} else {
				return true;
			}
		} else {
			return false;
		}
	}

	/**
	 * @Title: isFileSize
	 * @Description: 判断上传文件大小是否允许上传
	 * @param file
	 * @return boolean
	 */
	public static boolean isFileSize(MultipartFile file) {
		if (file != null) {
			long fileSize = file.getSize();
			// 判断上传文件大小是否允许上传
			if (fileSize <= FILE_SIZE_30M * 1024 * 1024) {
				return true;
			} else {
				return false;
			}
		} else {
			return false;
		}

	}

	/**
	 * @Title: attachmentUpload
	 * @Description: 文件上传使用参考方法
	 * @param request
	 * @param response
	 */
	public static void attachmentUpload(HttpServletRequest request,
			HttpServletResponse response) {
		CommonsMultipartResolver multipartResolver = new CommonsMultipartResolver();
		if (multipartResolver.isMultipart(request)) {
			MultipartHttpServletRequest multipartRequest = (MultipartHttpServletRequest) request;

			Iterator<String> iter = multipartRequest.getFileNames();
			while (iter.hasNext()) {
				MultipartFile file = multipartRequest.getFile((String) iter
						.next());
				// 判断上传文件类型是否允许上传
				if (!isAttachment(file)) {
					if (file != null) {
						String filename = "demo" + file.getOriginalFilename();
						String path = "D:/" + filename;
						File localfile = new File(path);
						try {
							file.transferTo(localfile);
						} catch (IllegalStateException | IOException e) {
							e.printStackTrace();
						}
					}
				}
			}
		}
	}

}
